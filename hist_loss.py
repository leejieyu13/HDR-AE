import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def gamma(hdrs, EVs = [1]):
    #with size [batch_size, n_channel, h, w]
    ldrs = []
    for i in range(hdrs.size(0)):
        hdr = hdrs[i] * (2**EVs[i])
        bpp = 12
        hdr_clip = torch.clamp(hdr, min=1e-12, max=2**bpp-1)
        hdr_norm = hdr_clip/(2**bpp-1)
        ldr = hdr_norm**(1/2.2)
        ldrs.append(torch.clamp(ldr, min= 0.0, max = 1.0))
    ldrs = torch.stack(ldrs)
    return ldrs

def rgb2gray(img):
    if len(img.size()) == 4:
        result = 0.299 * img[:,0,:,:] + 0.587 * img[:,1,:,:] + 0.114 * img[:,2,:,:] 
        result = result.unsqueeze(1)
    elif len(img.size()) == 3:
        result = 0.299 * img[0,:,:] + 0.587 * img[1,:,:] + 0.114 * img[2,:,:] 
        result = result.unsqueeze(0)
    else:
        result= None
    return result
    
def multiScaleHist(ldr, scales = [1,2,5], bins_num=128):
    N, _, H, W = ldr.size()
    hist_batch = [] 
    for i in range(N):
        hist_list = []
        hist = torch.histc(ldr.flatten(), bins_num, min=0.0, max=1.0)/ (H * W)
        hist_list.append(hist)
        h = int(H/scales[1])
        w = int(W/scales[1])
        for i in range(scales[1]):
            for j in range(scales[1]):
                hist =  torch.histc(ldr[:, :, i*h:(i+1)*h, j*w:(j+1)*w].flatten(), bins_num, min=0.0, max=1.0) /(h*w)
                hist_list.append(hist)
        # 7*7
        h = int(H/scales[2])
        w = int(W/scales[2])
        for i in range(scales[2]):
            for j in range(scales[2]):
                hist =  torch.histc(ldr[:, :, i*h:(i+1)*h, j*w:(j+1)*w].flatten(), bins_num, min=0.0, max=1.0) /(h*w)
                hist_list.append(hist)  
        hist_batch.append(torch.stack(hist_list, dim = 0))
    result = torch.stack(hist_batch, dim = 0)
    return result

       
    
class hist_loss(nn.Module):
    def __init__(self):
        super(hist_loss, self).__init__()
        self.bins = 128
        self.p_equal = torch.ones([1, self.bins], dtype=torch.float32)/self.bins

        if torch.cuda.is_available():
           self.p_equal = self.p_equal.cuda() 
    
    
    def hist_region(self, gray, block_size):
        _, H, W = gray.shape
        hs = int(H/block_size)
        ws = int(W/block_size)
        hist = []
        for i in range(block_size):
            for j in range(block_size):
                x = gray[0, i*hs:(i+1)*hs, j*ws:(j+1)*ws]
                p = histogram(x.flatten().unsqueeze(0), torch.torch.linspace(0.0, 1.0, self.bins), bandwidth=torch.tensor(0.005))
                hist.append(p)

        return hist
    
    def sim_to_ref_hist(self, hist, hist_ref):
        score = []
        for i, p in enumerate(hist):
            cos_hist =  F.cosine_similarity(p, hist_ref[i])
            loss = 1.0 - cos_hist      
            score.append(loss)
        return torch.mean(torch.stack(score))
    

    def forward(self, hdr, param, block_size = [1, 5]):
        n_sample, n_channel, _, _ = hdr.shape
        loss = []
        weights = [1.0, 1.0]
        for i in range(n_sample):
            l_temp = 0
            ldr = gamma(hdr[i], param[i])
            if n_channel == 3:            
                ldr = rgb2gray(ldr)
            
            for idx, block in enumerate(block_size):
                h = self.hist_region(ldr, block)         
                hist_ref = [self.p_equal] * block**2
                l = self.sim_to_ref_hist(h, hist_ref)
                l_temp += l * weights[idx] 
            loss.append(l_temp)
        return torch.mean(torch.stack(loss))


    def loss_ldr(self, ldr, block_size =  [1, 5]):
        n_channel, _, _ = ldr.shape
        loss = 0
        weights = [1.0, 1.0]

        if n_channel == 3:            
            gray = rgb2gray(ldr)
        else:
            gray = ldr
        for idx, block in enumerate(block_size):
            h = self.hist_region(gray, block)         
            hist_ref = [self.p_equal] * block**2
            l = self.sim_to_ref_hist(h, hist_ref)
            loss += l * weights[idx] 

        return loss
    
   


def marginal_pdf(values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor,
                 epsilon: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function that calculates the marginal probability distribution function of the input tensor
        based on the number of histogram bins.

    Args:
        values (torch.Tensor): shape [BxNx1].
        bins (torch.Tensor): shape [NUM_BINS].
        sigma (torch.Tensor): shape [1], gaussian smoothing factor.
        epsilon: (float), scalar, for numerical stability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
          - torch.Tensor: shape [BxN].
          - torch.Tensor: shape [BxNxNUM_BINS].

    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("Input values type is not a torch.Tensor. Got {}"
                        .format(type(values)))

    if not isinstance(bins, torch.Tensor):
        raise TypeError("Input bins type is not a torch.Tensor. Got {}"
                        .format(type(bins)))

    if not isinstance(sigma, torch.Tensor):
        raise TypeError("Input sigma type is not a torch.Tensor. Got {}"
                        .format(type(sigma)))

    if not values.dim() == 3:
        raise ValueError("Input values must be a of the shape BxNx1."
                         " Got {}".format(values.shape))

    if not bins.dim() == 1:
        raise ValueError("Input bins must be a of the shape NUM_BINS"
                         " Got {}".format(bins.shape))

    if not sigma.dim() == 0:
        raise ValueError("Input sigma must be a of the shape 1"
                         " Got {}".format(sigma.shape))

    bins = bins.to(values.device)
    sigma = sigma.to(values.device)
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return (pdf, kernel_values)



def histogram(x: torch.Tensor, bins: torch.Tensor, bandwidth: torch.Tensor,
              epsilon: float = 1e-10) -> torch.Tensor:
    """Function that estimates the histogram of the input tensor.

    The calculation uses kernel density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x (torch.Tensor): Input tensor to compute the histogram with shape :math:`(B, D)`.
        bins (torch.Tensor): The number of bins to use the histogram :math:`(N_{bins})`.
        bandwidth (torch.Tensor): Gaussian smoothing factor with shape shape [1].
        epsilon (float): A scalar, for numerical stability. Default: 1e-10.

    Returns:
        torch.Tensor: Computed histogram of shape :math:`(B, N_{bins})`.

    Examples:
        >>> x = torch.rand(1, 10)
        >>> bins = torch.torch.linspace(0, 255, 128)
        >>> hist = histogram(x, bins, bandwidth=torch.tensor(0.9))
        >>> hist.shape
        torch.Size([1, 128])
    """

    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)

    return pdf
    
              


