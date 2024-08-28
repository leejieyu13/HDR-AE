import cv2
import os
import glob
import hist_loss
import torch
import argparse


def max_metric(imgs, loss, block_size, weights):
    max_l = 0
    for i, img in enumerate(imgs):
        img = (torch.from_numpy(img.transpose(2, 0, 1))/255)#.cuda()
        l = loss.ldr_loss(img, block_size, weights=weights, include_sky = False, diff_flag = False)
        if l >= max_l:
            max_l = l
            best_indx = i
    return best_indx    


def main(dataset_dir):
    root_dirs = []
    for folder in next(os.walk(dataset_dir))[1]:
        root_dirs.append(folder)
    root_dirs = sorted(root_dirs)
    weights = [0.2, 0.8]
    block_size = [1,3]
    output_file = dataset_dir + '/selected_image_info.txt'
    fout = open(output_file,'w')
    loss = hist_loss.hist_loss()
    for i in range(len(root_dirs)):    
        print(root_dirs[i])
        ## left
        left_dir = dataset_dir + '/' + root_dirs[i] + '/left'
        img1s = []
        filenames1 = []
        for filename in glob.glob(os.path.join(left_dir, '*.jpg')):
            img = cv2.imread(filename, -1)
            img1s.append(img)
            filenames1.append(filename.split('\\')[-1])
        
        ## right
        right_dir = dataset_dir + '/' + root_dirs[i] + '/right'
        img2s = []
        filenames2 = []

        for filename in glob.glob(os.path.join(right_dir, '*.jpg')):
            img = cv2.imread(filename, -1)
            img2s.append(img)
            filenames2.append(filename.split('\\')[-1])

        s1 = max_metric(img1s, loss, block_size, weights)
        fout.write( filenames1[s1] + '\n')
        s2 = max_metric(img2s, loss, block_size, weights)
        fout.write( filenames2[s2] + '\n')
        print(filenames1[s1])
        print(filenames2[s2])
        break
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str)  
    args = parser.parse_args()
    main(args.dataset_dir)

    