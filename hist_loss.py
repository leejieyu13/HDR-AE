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
        # ldr = torch.clamp(hdr_norm**(1/2.2), min=0.0, max = 1.0)
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
        h = int(H/scales[2])
        w = int(W/scales[2])
        for i in range(scales[2]):
            for j in range(scales[2]):
                hist =  torch.histc(ldr[:, :, i*h:(i+1)*h, j*w:(j+1)*w].flatten(), bins_num, min=0.0, max=1.0) /(h*w)
                hist_list.append(hist)  
                
        hist_batch.append(torch.stack(hist_list, dim = 0))
    result = torch.stack(hist_batch, dim = 0)
    return result

class HistLayer(nn.Module):
    """Deep Neural Network Layer for Computing Differentiable Histogram.
    Computes a differentiable histogram using a hard-binning operation implemented using
    CNN layers as desribed in `"Differentiable Histogram with Hard-Binning"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Attributes:
        in_channel (int): Number of image input channels.
        numBins (int): Number of histogram bins.
        learnable (bool): Flag to determine whether histogram bin widths and centers are
            learnable.
        centers (List[float]): Histogram centers.
        widths (List[float]): Histogram widths.
        two_d (bool): Flag to return flattened or 2D histogram.
        bin_centers_conv (nn.Module): 2D CNN layer with weight=1 and bias=`centers`.
        bin_widths_conv (nn.Module): 2D CNN layer with weight=-1 and bias=`width`.
        threshold (nn.Module): DNN layer for performing hard-binning.
        hist_pool (nn.Module): Pooling layer.
    """

    def __init__(self, in_channels=1, num_bins=128):
        super(HistLayer, self).__init__()

        # histogram data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.learnable = False
        bin_edges = np.linspace(0, 1, num_bins + 1)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2
        self.centers = centers[:-1]
        self.width = (bin_edges[2] - bin_edges[1]) / 2

        # prepare NN layers for histogram computation
        self.bin_centers_conv = nn.Conv2d(
            self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.in_channels,
            bias=True,
        )
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_centers_conv.bias.data = torch.nn.Parameter(
            -torch.tensor(self.centers, dtype=torch.float32)
        )
        self.bin_centers_conv.bias.requires_grad = self.learnable

        self.bin_widths_conv = nn.Conv2d(
            self.numBins * self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.numBins * self.in_channels,
            bias=True,
        )
        self.bin_widths_conv.weight.data.fill_(-1)
        self.bin_widths_conv.weight.requires_grad = False
        self.bin_widths_conv.bias.data.fill_(self.width)
        self.bin_widths_conv.bias.requires_grad = self.learnable

        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight
        self.threshold = nn.Threshold(1, 0)
        self.hist_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image):
        """Computes differentiable histogram.
        Args:
            input_image: input image.
        Returns:
            flattened and un-flattened histogram.
        """
        # |x_i - u_k|
        xx = self.bin_centers_conv(input_image)
        xx = torch.abs(xx)

        # w_k - |x_i - u_k|
        xx = self.bin_widths_conv(xx)

        # 1.01^(w_k - |x_i - u_k|)
        xx = torch.pow(torch.empty_like(xx).fill_(1.01), xx)

        # Φ(1.01^(w_k - |x_i - u_k|), 1, 0)
        xx = self.threshold(xx)

        # clean-up
        xx = self.hist_pool(xx)
        one_d = torch.flatten(xx, 1)
        return one_d
       
    
class hist_loss(nn.Module):
    def __init__(self):
        super(hist_loss, self).__init__()
        self.bins = 128
        self.p_equal = torch.ones([1, self.bins], dtype=torch.float32)/self.bins

        if torch.cuda.is_available():
           self.p_equal = self.p_equal.cuda() 
    
    
    def ldr_loss(self, ldr, block_size = [1,3], weights = [0.2, 0.8], include_sky = False, diff_flag = True):
        n_channel, _, _ = ldr.shape
        loss = 0

        if n_channel == 3:            
            gray = rgb2gray(ldr)
        else:
            gray = ldr

        for idx, block in enumerate(block_size):
            h = self.hist_region(gray, block, include_sky, diff_flag)         
            hist_ref = [self.p_equal] * block**2
            l = self.sim_to_ref_hist(h, hist_ref)
            loss += l * weights[idx]
        return loss/(sum(weights))


    def hist_region(self, gray, block_size, include_sky = False, diff_flag = True):
        _, H, W = gray.shape
        hs = int(H/block_size)
        ws = int(W/block_size)
        hist = []
        for i in range(block_size):
            if not include_sky and block_size>1 and i == 0:
                continue
            for j in range(block_size):
                x = gray[0, i*hs:(i+1)*hs, j*ws:(j+1)*ws]
                if diff_flag:
                    p = histogram(x.flatten().unsqueeze(0), torch.torch.linspace(0.0, 1.0, self.bins), bandwidth=torch.tensor(0.005))
                else:
                    p = torch.histc(x.flatten().unsqueeze(0), self.bins, min=0.0, max=1.0)/(H*W)
                hist.append(p)
        return hist
    
    def sim_to_ref_hist(self, hist, hist_ref):
        score = []
        for i, p in enumerate(hist):
            cos_hist =  F.cosine_similarity(p, hist_ref[i])
            score.append(cos_hist)
        return torch.mean(torch.stack(score))
    

    def forward(self, hdr, param, block_size = [1, 5], weights = [0.2, 0.8]):
        n_sample, n_channel, _, _ = hdr.shape
        loss = []
        ldrs = gamma(hdr, param)
        if n_channel == 3:            
            ldrs = rgb2gray(ldrs)
        for ldr in ldrs:
            loss.append(self.ldr_loss(ldr, block_size, weights))
        return torch.mean(torch.stack(loss))


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