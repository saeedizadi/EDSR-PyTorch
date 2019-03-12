import torch.nn as nn
from PIL import Image
import torch
import torchvision.transforms as transforms
import  numpy as np


def make_model(args, parent=False):
    return BICUBIC(args)


class BICUBIC(nn.Module):
    def __init__(self, args):
        super(BICUBIC, self).__init__()

        self.scale = float(args.scale[0])
        self.dummy = nn.Conv2d(1, 1, 3)
        self.resize = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=1024, interpolation=Image.BICUBIC), transforms.ToTensor()])

    def forward(self, x):
        return self.interpolate(x)

    def interpolate(self, tensor):
        im = Image.fromarray(tensor.squeeze().cpu().data.numpy())
        h, w = im.size
        nw, nh = int(h * self.scale), int(w * self.scale)
        im = im.resize(size=(nw, nh), resample=Image.BICUBIC)
        temp = torch.from_numpy(np.array(im, dtype=np.float32)).cuda()
        return temp

