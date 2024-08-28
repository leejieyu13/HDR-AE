import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class ToTensor(object):
    """ Convert ndarrays in sample to Tensor"""

    def __init__(self, opts):
        self.sequence = opts.sequence

    def __call__(self, sample):
        # tens = transforms.ToTensor()

        img_seq, target = sample['img_seq'], sample['target']#, sample['img_name'] 

        for i in range(len(img_seq)):
            img_seq[i] = torch.from_numpy(img_seq[i].transpose(2, 0, 1))
        target_torch = torch.FloatTensor(target)
        return {'img_seq': img_seq, 'target': torch.FloatTensor(target)}#, 'img_name': name}

class Resize(object):
    """ Convert the image to have a specified size """

    def __init__(self, output_size, opts):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.sequence = opts.sequence

    def __call__(self, sample):

        resize = transforms.Resize(self.output_size)

        img_seq, target = sample['img_seq'], sample['target']#, sample['img_name'] 
        for i in range(len(img_seq)):
            img_seq[i] = resize(img_seq)

        return {'img_seq': img_seq, 'target':target}#, 'img_name': name}

class HorizontalFlip(object):
    """ Randomly flip the images horizontally. Perform the same operation to all three images. """

    def __init__(self, opts):
        self.sequence = opts.sequence

    def __call__(self, sample):

        randomflip = transforms.RandomHorizontalFlip()
        flip = transforms.RandomHorizontalFlip(p=1.0)

        img_seq, target = sample['img_seq'], sample['target']#, sample['img_name'] 
    

        # randomly flip first image, if flipped, flip the other two

        img0 = randomflip(img_seq[0])
        if img0 != img_seq[0]:
            for i in range(len(img_seq)):
                img_seq[i] = flip(img_seq)


        return {'img_seq': img_seq, 'target': target}#, 'img_name': name}

class VerticalFlip(object):
    """ Randomly flip the images vertically. Perform the same operation to all three images. """

    def __init__(self, opts):
        self.sequence = opts.sequence

    def __call__(self, sample):

        randomflip = transforms.RandomVerticalFlip()
        flip = transforms.RandomVerticalFlip(p=1.0)

     
        img_seq, target = sample['img_seq'], sample['target']#, sample['img_name'] 
    

        # randomly flip first image, if flipped, flip the other two
        img0 = randomflip(img_seq[0])
        if img0 != img_seq[0]:
            for i in range(len(img_seq)):
                img_seq[i] = flip(img_seq)


        return {'img_seq': img_seq, 'target': target}#, 'img_name': name}



