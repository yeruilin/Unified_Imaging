import os 
import glob
import math
from pickle import TRUE
import numpy as np
import cv2
from torch.utils.data import Dataset,DataLoader
import scipy.io as sio
import random
import torch
import torch.nn.functional as F


## yrl：使用alpha数据集训练NLOST网络
def NLOSDatasetFileList(root_path,fortrain=True):
    mea = []
    im = []
    de =[]
    inten=[]
    test_mea = []
    test_im = []
    test_de =[]
    test_inten=[]
    
    for dir in root_path:
        pathlist = glob.glob('%s/confocal-*.hdr'% (dir))
        num=len(pathlist)
        train_num=math.floor(0.99*num)

        for i in range(train_num):
            path = glob.glob('%s/confocal-%d.hdr'% (dir,i))
            mea.append(path[0])
            # path = glob.glob('%s/phasor-%d.hdr'% (dir,i))
            # im.append(path[0])
            path = glob.glob('%s/depth-%d.hdr'% (dir,i))
            de.append(path[0])
            path = glob.glob('%s/intensity-%d.hdr'% (dir,i))
            inten.append(path[0])
        
        for i in range(train_num,num):
            path = glob.glob('%s/confocal-%d.hdr'% (dir,i))
            test_mea.append(path[0])
            # path = glob.glob('%s/phasor-%d.hdr'% (dir,i))
            # test_im.append(path[0])
            path = glob.glob('%s/depth-%d.hdr'% (dir,i))
            test_de.append(path[0])
            path = glob.glob('%s/intensity-%d.hdr'% (dir,i))
            test_inten.append(path[0])
                    
    train_sample = {'Mea': mea, 'dep': de, 'inten':inten}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'inten':test_inten}

    if fortrain:
        return train_sample
    else:
        return test_sample
    
def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)


class NLOSPoissonNoise:

    def __init__(self, background=[0.05, 0.5]):
        self.rate = background

    def __call__(self, x):
        if isinstance(self.rate, (int, float)):
            rate = self.rate
        elif isinstance(self.rate, (list, tuple)):
            rate = random.random() * (self.rate[1] - self.rate[0]) 
            rate += self.rate[0]
        poisson = torch.distributions.Poisson(rate)
        # shot noise + background noise
        x = torch.poisson(x) + poisson.sample(x.shape).cuda()
        return x

    def __repr__(self):
        return 'Introduce shot noise and background noise to raw histograms'


class NLOSRandomScale:

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, x):
        if isinstance(self.scale, (int, float)):
            x *= self.scale
        elif isinstance(self.scale, (list, tuple)):
            scale = random.random() * (self.scale[1] - self.scale[0]) 
            scale += self.scale[0]
            x *= scale
        return x

    def __repr__(self):
        return 'Randomly scale raw histograms'

class NLOSCompose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        repr_str = ''
        for t in self.transforms:
            repr_str += t.__repr__() + '\n'
        return repr_str


def get_transform(scale=1, background=0):
    transform = [NLOSRandomScale(scale)]
    if background != 0:
        transform += [NLOSPoissonNoise(background)]
    transform = NLOSCompose(transform)
    return transform

## 包含三维数据的数据集
class NLOSDataset(Dataset):
    def __init__(
        self, 
        root,               # dataset root directory
        split=True,              # data split ('train', 'val')
        target_size=256,    # target image size (unit: px)
        clip=512,           # time range of histograms
        background=0,       # background noise rate (float or float tuple)
        target_noise=0,     # standard deviation of target image noise
    ):
        super(NLOSDataset, self).__init__()

        self.root = root
        self.target_size = target_size # N
        self.clip = clip # M
        self.transform = get_transform(scale=1.0, background=background)
        
        self.target_noise = target_noise

        self.split = split
        self.data_list = NLOSDatasetFileList(self.root,self.split)
        # print(self.split,len(self.data_list['Mea']))


    def _load_meas(self, idx):
        path = self.data_list['Mea'][idx]
        # print(path)
        #check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) ## 三通道合并为单通道
                x = x.astype(np.float32)
                width=int(np.sqrt(x.shape[0]))
                x = x.reshape(width,width, x.shape[1], 1)
                x = x.transpose(3, 2, 0, 1)
                
            x = x[:, :self.clip]                                # (1/3, t, h, w)

        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))
        
        x = torch.from_numpy(x.astype(np.float32))              # (1/3, t, h, w)
        x=x/(torch.max(x) + 1e-8) # Normalization

        return x
    
    def _load_depth(self, idx):
        path = self.data_list['dep'][idx]
        #check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # x = cv2.resize(x, (self.target_size, self.target_size))
                x = x / (np.max(x) + 1e-8)
                # x = x[..., 0] # 取第一通道作为深度图，这里我们需要三通道输入flux encoder
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))              # (h, w, c)
        x = x.permute(2, 0, 1) # (c, h, w) 
        return x
    
    def _load_intensity(self, idx):
        path = self.data_list['inten'][idx]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                # x = cv2.resize(x, (self.target_size, self.target_size))
                x = x.astype(np.float32)
                if ext == 'hdr':
                    x = x / (np.max(x) + 1e-8)
                else:
                    x = x / 255

        except:
            raise ValueError('image loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))             
        x = x.permute(2, 0, 1)                               # ( 1/3, h, w)       #  1 256 256 
        if self.target_noise > 0:
            x += torch.rand_like(x) * self.target_noise
            x = torch.clamp(x, min=0)
        return x

    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas = self._load_meas(idx) # 不加噪
        # meas =self.transform(self._load_meas(idx)) # 加噪
        depths = self._load_depth(idx)
        intensity=self._load_intensity(idx)
        sample = {'meas': meas, 'dep': depths, 'inten': intensity}
        return sample
    
if __name__=="__main__":
    
    dataset2 = NLOSDataset(['/data/yrl/Flux.1/NLOS_data0310/'],split=True,target_size=128)
    print(len(dataset2))
    data=dataset2[0]
    print(data["dep"].shape) # [1,imgsize1,imgsize1]
    print(data["inten"].shape) # [3,imgsize1,imgsize1]
    print(data["meas"].shape) # [1,timebin,imgsize,imgsize]

