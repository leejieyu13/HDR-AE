import torch
import torch.nn
import torch
import numpy as np

import math

def print_network(net):
	num_params = sum([param.numel() for param in net.parameters()])
	print(net)
	print('Total number of parameters: {}'.format(num_params))

def gaussian2d(u, v, sigma):
    pi = 3.1416
    intensity = 1 / (2.0 * pi * sigma * sigma) * math.exp(- 1 / 2.0 * ((u ** 2) + (v ** 2)) / (sigma ** 2))
    return intensity

def gaussianKernal(r, sigma):
    kernal = np.zeros([r, r])
    center = (r - 1) / 2.0
    for i in range(r):
        for j in range(r):
            kernal[i, j] = gaussian2d(i - center, j - center, sigma)
    kernal /= np.sum(np.sum(kernal))
    return kernal

def weights_init_Gaussian_blur(sigma=1.0):
    def sub_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = m.weight.data.size()
            gaussian_blur = gaussianKernal(weight_shape[2], sigma)
            for i in range(weight_shape[0]):
                m.weight.data[i, 0, :, :] = torch.from_numpy(gaussian_blur)
            if not m.bias is None:
                m.bias.data.zero_()
    return sub_func

def weights_init_He_normal(m):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Transpose') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
        m.weight.data.normal_(0.0, std)
        # init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        if not m.bias is None:
            m.bias.data.zero_()

def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=4, scale_lr=10.0):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr


def concatenate_dicts(*dicts):
	concat_dict = {} # Create a dictionary
	for key in dicts[0]: 
		concat_dict[key] = [] # loop through the keys in the first dictionary, assign to new dictionary
		for d in dicts:
			val = d[key] # loop through each dictionary, the val var is the value of the key in each of the dictionaries
			if isinstance(val, list):
				concat_dict[key] = concat_dict[key] + val # if val is a list, append to the new dictionary
			else:
				concat_dict[key].append(val) # if there is only once instance, add to dict

	return concat_dict

def compute_dict_avg(dict):
	avg_dict = {}
	for key, val in dict.items():
		avg_dict[key] = np.mean(np.array(val))
	return avg_dict

def tag_dict_keys(dict, tag):
	new_dict = {}
	for key, val in dict.items():
		new_key = key + '/' + tag
		new_dict[new_key] = val
	return new_dict

