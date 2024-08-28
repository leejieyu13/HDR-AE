import numpy as np
import hist_loss
import torch

def initial(hdr, EV0=None):
    Alpha = 1.7   # Alpha decides the initial simplex size
    
    ## 1. Construct initial simplex
    # Generate random initial point
    Init_Points = torch.zeros((2,1)).to(hdr.device)
    if EV0 == None:
        Init_Points[0,:] = torch.from_numpy(np.random.rand(1) * (10+6) + (-6))
    else:
        Init_Points[0,:] = EV0
    # Read initial image based on random generated point
    
    # Decide exporing points based on a intensity of current captured image.
    ldr = hist_loss.rgb2gray(hist_loss.gamma(hdr, Init_Points[0])) * 255
    Intensity = torch.mean(ldr)
    
    
    if Intensity >= 128:
        h = -1/Alpha*(Intensity/255)
    else:
        h = Alpha*(1 - Intensity/255)
    
    Init_Points[1,:] = Init_Points[0]*(1 + h)
    # print('Initial data:', Init_Points)
    
    return Init_Points


def nelder_mead_AE(x, loss_function, hdr, EV0 = None, reg_weight =0.01, max_feval = 50):
    ## x is the simplex points
    ## 0. Define algorithm constants
    rho = 1;    # rho > 0, reflection coefficient
    xi  = 2;   # xi  > max(rho, 1), expansion coefficient
    gam = 0.5;  # 0 < gam < 1,  contraction coefficient
    sig = 0.5;  # 0 < sig < 1, shrink coefficient

    tolerance = 1.0E-06  # end criteria
    _, n_dim = x.size()
    # Evaluate each point of initial simplex
    f = []
    
    for param in x:
        loss = loss_function(hdr, param) 
        f.append(loss) 
    f = torch.stack(f)
    n_feval = n_dim + 1
    f, indx = torch.sort(f)
    x = x[indx]
    # Begin the Nelder Mead iteration.
    converged = 0
    diverged  = 0
    
    while not converged and not diverged:
        ##x = [[a],[b]]
        #  1. Compute the midpoint of the simplex opposite the worst point.
        x_bar = torch.sum(x[0:n_dim,:])/n_dim

        #  2. Compute the reflection point.
        x_r = ( 1 + rho ) * x_bar -  rho * x[n_dim,:]
        f_r = loss_function(hdr, x_r) 
        n_feval = n_feval + 1

        #  3-1. Accept the reflection point
        if f[0] <= f_r and f_r <= f[n_dim]:
            x[n_dim,:] = x_r
            f[n_dim] = f_r
        #  3-2. Test for possible expansion.
        elif f_r < f[0]:
            x_e = ( 1 + rho * xi ) * x_bar - rho * xi * x[n_dim,:]
            f_e  = loss_function(hdr, x_e) 
            n_feval = n_feval + 1
            #  Can we accept the expanded point?
            if f_e < f_r:
                x[n_dim,:] = x_e
                f[n_dim] = f_e
                
            else:
                x[n_dim,:] = x_r
                f[n_dim] = f_r
            
        #  3-3.Outside contraction.
        elif f[n_dim-1] <= f_r and f_r < f[n_dim]:
            x_c = (1 + rho * gam) * x_bar - rho * gam * x[n_dim,:]
            f_c  = loss_function(hdr, x_c) 
            n_feval = n_feval+1
            
            if f_c <= f_r: # accept the contracted point
                x[n_dim,:] = x_c
                f[n_dim] = f_c
               
            else:
               x, f = shrink(x, f, hdr, sig, loss_function, reg_weight)
               n_feval = n_feval+n_dim
            
        #  3-4. Intra contraction.
        else:
            x_c = ( 1 - gam ) * x_bar + gam   * x[n_dim,:]
            f_c  = loss_function(hdr, x_c) 
            n_feval = n_feval+1

            #  Can we accept the contracted point?
            if f_c < f[n_dim]:
                x[n_dim,:] = x_c
                f[n_dim] = f_c
          
            else:
                x, f = shrink(x, f, hdr, sig, loss_function, EV0, reg_weight)
                n_feval = n_feval+n_dim
          
        #  Resort the points.  Note that we are not implementing the usual
        #  Nelder-Mead tie-breaking rules  (when f(1) = f(2) or f(n_dim) =
        #  f(n_dim+1)...

        f, indx = torch.sort(f)
        x = x[indx]

        #  Test for convergence
        converged = f[n_dim] - f[0] < tolerance
        # print('x', x)
        # print('f', f)

        #  Test for divergence
        diverged = max_feval < n_feval


    x_opt = x[0,:]
    # if diverged:
    #     print ('NELDER_MEAD - Warning!\n')
    #     print('  The maximum number of function evaluations was exceeded\n')
    #     print('  without convergence being achieved.\n' )
     
     

    return x_opt, n_feval



def shrink( x, f, hdr, sig, loss_function, EV0=None, reg_weight=0.01):
    n_feval, n_dim = x.size()

    for i in range(1, n_feval):
        x[i,:] = sig * x[i,:] + (1.0-sig)*x[0,:]
        f[i] = loss_function(hdr, x[i,:]) 
        # f[i] = f[i] + reg_weight*(x[i,:]-EV0)**2
  
    return  x, f



