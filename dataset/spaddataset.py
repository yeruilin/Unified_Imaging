# The SPAD data pre-process function
import torch
import torch.utils.data
import scipy.io
import numpy as np

from skimage.transform import resize
import math

def find_files(directory,ext):
    obj_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                obj_files.append(os.path.join(root, file))
    obj_files=sorted(obj_files) # 按照名称的字母顺序排序
    return obj_files

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        rates, spad, bins_hr, bins = sample['gt'],\
                                                sample['meas'],\
                                                sample['bins_hr'],\
                                                sample['dep']
        sbr, photons = sample['sbr'], sample['photons']

        rates = torch.from_numpy(rates)
        spad = torch.from_numpy(spad)
        bins_hr = torch.from_numpy(bins_hr)
        bins = torch.from_numpy(bins)
        return {'gt': rates, 'meas': spad,
                'bins_hr': bins_hr, 'dep': bins,
                'sbr': sbr, 'photons': photons}


class SpadDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, timebin=512,imgsize=64, mode='train', ratio=0.8, transform=None):
        """__init__
        :param datapath: path to text file with list of
                        training files (intensity files)
        :param transform: transform (callable, optional):
                        Optional transform to be applied
                        on a sample.
        """

        self.transform = transform
        self.timebin=timebin
        self.imgsize=imgsize

        files=find_files(datapath,"mat")
        if mode=='train':
            start=0
            end=int(ratio*len(files))
        elif mode=='val':
            start=int(ratio*len(files))+1
            end=len(files)

        self.spad_files=files[start:end]

    def __len__(self):
        return len(self.spad_files)

    def tryitem(self, idx):
        # simulated spad measurements
        spad = np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(self.spad_files[idx])['spad'])) # [64*64,1024]
        spad=resize(spad/np.max(spad),(self.imgsize*self.imgsize,self.timebin)).reshape([1,self.imgsize,self.imgsize,self.timebin])
        spad = np.transpose(spad, (0, 3, 2, 1))

        # normalized pulse
        rates = np.asarray(scipy.io.loadmat(self.spad_files[idx])['rates']) # [64,64,1024]
        rates=resize(rates/np.max(rates),(self.imgsize,self.imgsize,self.timebin)).reshape([1,self.imgsize,self.imgsize,self.timebin])
        rates = np.transpose(rates, (0, 3, 1, 2))
        
        rates = rates / np.sum(rates, axis=1)[None, :, :, :] # [1,timebin,imgsize,imgsize]
        rates=rates*(rates>0.005)

        sample = {'gt': torch.from_numpy(rates), 'meas': torch.from_numpy(spad)}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        # sample=torch.from_numpy(sample).float()
        return sample

if __name__=="__main__":
    datapath="../../Singlephoton_data/"
    dataset1=SpadDataset(datapath,timebin=512,imgsize=64)
    print(len(dataset1))
    print(dataset1[0]["gt"].shape) # [1,timebin,imgsize,imgsize]
    print(dataset1[0]["meas"].shape) # [1,timebin,imgsize,imgsize]
    print('data1max!!!',dataset1[0]["gt"].max())