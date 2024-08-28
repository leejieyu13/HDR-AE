import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
import argparse

K1 = np.array([[814.889003, -0.384418, 764.659138], [0.000000, 814.206990, 576.541219], [0.000000, 0.000000, 1.000000]])
K2 = np.array([[816.254045, -0.516877, 767.769055], [0.000000, 815.958860, 580.307083], [0.000000, 0.000000, 1.000000]])
D1 = np.array([-0.055030, 0.122773, 0.001917, -0.001426, -0.065038])
D2 = np.array([-0.052789, 0.123278, 0.000337, -0.001296, -0.067356])
R = np.array([[0.999887, -0.004519, -0.014343], [0.004515,  0.999990, -0.000323], [0.014345,  0.000259,  0.999897]])
T = np.array([-0.201597, -0.001746, 0.000769])
baseline = np.linalg.norm(T)
width = 1600
height = 1200

def hist2d(x,y, roi):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    hist, xbins, ybins, im = ax.hist2d(x, y, bins=10, range = roi)
    plt.close('all')
    return np.sum(hist>0)
 
def shi_tomasi(img1,img2, n_feature, ransac_flag = True):
    feature_params = dict(maxCorners=n_feature, qualityLevel=0.01, minDistance=10)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    features = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    # Calculate optical flow
    features_r, status, error = cv2.calcOpticalFlowPyrLK(img1, img2, features, None, **lk_params)
    # Select good points
    features_l = features[(status == 1) & (error <= 30)]
    features_r = features_r[(status == 1) & (error <= 30)]
    pts_l_norm  = cv2.undistortPoints(features_l, cameraMatrix = K1, distCoeffs = D1)
    pts_r_norm  = cv2.undistortPoints(features_r, cameraMatrix = K2, distCoeffs = D2)
    # Calculate the essential matrix
    if ransac_flag:
        E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.RANSAC, prob=0.999, threshold=1e-3)       
    else:
        E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.LMEDS)
    # mask = mask.flatten()
    # features_l = features_l[mask>0]
    # features_r = features_r[mask>0]
    # Recover the relative camera pose
    _, R_est, t_est, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
    t_est = t_est*baseline/np.linalg.norm(t_est) # the length of baseline is given
    error_z1 = np.sqrt((t_est[0]-T[0])**2+(t_est[1]-T[1])**2+(t_est[2]-T[2])**2)
    error_z2 = np.sqrt((t_est[0]+T[0])**2+(t_est[1]+T[1])**2+(t_est[2]+T[2])**2)
    error_z = min(error_z1, error_z2)   
    error_degree = np.degrees(np.arccos(0.5*np.trace(R.transpose()*R_est) - 0.5))
    if error_degree > 90:
        error_degree = abs(180 - error_degree) 
    return error_z[0], error_degree, features_l.reshape(-1,1,2), features_r.reshape(-1,1,2)

def match_orb(img1, img2, n_feature, ransac_flag):
    orb = cv2.ORB_create(nfeatures=n_feature)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is not None and des2 is not None:
        orb_matches = bf.match(des1, des2)
        kp1_coords = np.asarray([kp1[m.queryIdx].pt for m in orb_matches]).reshape(-1,1,2)
        kp2_coords = np.asarray([kp2[m.trainIdx].pt for m in orb_matches]).reshape(-1,1,2)
    
        pts_l_norm  = cv2.undistortPoints(kp1_coords, cameraMatrix = K1, distCoeffs = D1)
        pts_r_norm  = cv2.undistortPoints(kp2_coords, cameraMatrix = K2, distCoeffs = D2)
        
        if ransac_flag:
            E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.RANSAC, prob=0.999, threshold=1e-3)
        else:
            E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.LMEDS)

        points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
        
        t_est = t_est*baseline/np.linalg.norm(t_est) # the length of baseline is given
        # inliers = [orb_matches[i] for (i, good) in enumerate(mask) if good >0]
    
        t_est = t_est.reshape(-1)
        t_est2 = -t_est
        error_z1 = np.linalg.norm(t_est - T)
        error_z2 = np.linalg.norm(t_est2 - T)
        error_z = min(error_z1, error_z2)
        
        trace_value = 0.5 * (np.trace(R_est.T @ R) - 1)
        trace_value = np.clip(trace_value, -1.0, 1.0)  # Ensure value is within valid range for arccos
        
        error_degree = np.degrees(np.arccos(trace_value))
        if error_degree > 90:
            error_degree = abs(180 - error_degree)
 
    return error_z, error_degree, kp1_coords, kp2_coords


def main(selected_img_file, orb_flag, ransac_flag, output_folder, n_feature=1000):
    file_dict = dict()
    fin = open(selected_img_file)
    lines = fin.readlines()
    fin.close()
    for i in range(0,len(lines),2):
        left_file = lines[i].strip()
        right_file = lines[i+1].strip()
        file_name = lines[i].split('/')[-3]
        file_dict[file_name]=(left_file,right_file)
    for file_name, files in file_dict.items():
        result_dict = []        
        os.makedirs(output_folder,exist_ok=True)
        os.makedirs(os.path.join(output_folder, "left"),exist_ok=True)
        os.makedirs(os.path.join(output_folder, "right"),exist_ok=True)

        left_file = files[0]    
        right_file = files[1]

        img1 = cv2.imread(left_file,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)
        if orb_flag:
            error_z, error_degree, kp1_coords,kp2_coords = match_orb(img1, img2, n_feature, ransac_flag)
        else:
            error_z, error_degree, kp1_coords,kp2_coords = shi_tomasi(img1, img2, n_feature, ransac_flag)

        plt.imshow(img1,cmap='gray')  
        plt.plot(kp1_coords[:,0,0], kp1_coords[:,0,1], marker='.', markersize=5, color=(0, 1, 0), linestyle='None')
        plt.axis('off')
        plt.savefig(os.path.join(output_folder,"left", f'{file_name}.jpg'), dpi=300, bbox_inches='tight')  # dpi controls resolution, bbox_inches='tight' removes extra whitespace
        plt.close()

        plt.imshow(img2,cmap='gray')  
        plt.plot(kp2_coords[:,0,0], kp2_coords[:,0,1], marker='.', markersize=5, color=(0, 1, 0), linestyle='None')
        plt.axis('off')
        plt.savefig(os.path.join(output_folder,"right",f'{file_name}.jpg'), dpi=300, bbox_inches='tight')  # dpi controls resolution, bbox_inches='tight' removes extra whitespace
        plt.close()

        num_block_left = hist2d(kp1_coords[:,0,0],kp1_coords[:,0,1], [[0, width], [0, height]])
        num_block_right = hist2d(kp2_coords[:,0,0],kp2_coords[:,0,1], [[0, width], [0, height]])
        result_dict.append({
            'file_name': file_name,
            "rotation": str(error_degree),
            "translation": str(error_z*100),
            "num_block_left": str(num_block_left),
            "num_block_right": str(num_block_right),
            "num_matched_features":str(len(kp1_coords)),
            "left":left_file,
            "right":right_file,
            })
        
    csv_file = os.path.join(output_folder, f'result.csv')
    fieldnames = result_dict[0].keys()

    # Write list of dictionaries to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-features', type=int, default=1000)  
    parser.add_argument('--ransac', action='store_true', help='set RANSAC, ow LMedS')
    parser.add_argument('--orb', action='store_true', help='use ORB features, ow Shi-Tomasi features')
    parser.add_argument('--selected-result', type=str, help='output txt file from function select_images')
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
    selected_img_file = args.selected_result
    orb_flag = args.orb
    ransac_flag = args.ransac
    output_folder = args.output_dir
    selected_img_file = args.selected_result
    n_feature = args.num_features
    main(selected_img_file, orb_flag, ransac_flag, output_folder, n_feature)