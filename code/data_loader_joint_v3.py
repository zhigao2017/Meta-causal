''' Digit 实验
'''
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10, STL10, USPS

import os
import pickle
import numpy as np
import h5py
import cv2
from scipy.io import loadmat
from PIL import Image

from tools.autoaugment import SVHNPolicy, CIFAR10Policy
from tools.randaugment import RandAugment
from tools.causalaugment_v3 import RandAugment_incausal, FactualAugment_incausal, CounterfactualAugment_incausal, MultiCounterfactualAugment_incausal

class myTensorDataset(Dataset):
    def __init__(self, x, y, transform=None, transform2=None, transform3=None, twox=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.twox = twox
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        c, h, w =x.shape
        # print("x.shape:",x.shape)
        if self.transform is not None:
            x_RA = self.transform(x)
            # print("x_RA.shape:",x_RA.shape)
            if self.transform3 is not None:
                x_CA = self.transform3(x_RA)
                x_CA = x_CA.reshape(-1,c,h,w)
                # print("x_CA.shape:",x_CA.shape)           
                if self.transform2 is not None:
                    x_FA = self.transform2(x)
                    # x_FA = x_FA.view(c,13,h,w)
                    x_FA = x_FA.reshape(-1,c,h,w)
                    # print("x_FA_in getitem.shape:",x_FA.shape)
                    # print("x_FA.shape:",x_FA.shape)
                    return (x, x_RA, x_FA, x_CA), y
                else:
                    return (x, x_RA, x_CA), y
            else:
                if self.transform2 is not None:
                    x_FA = self.transform2(x)
                    x_FA = x_FA.reshape(-1,c,h,w)
                    return (x, x_RA, x_FA), y
                else:
                    if self.twox:
                        return (x, x_RA), y
                    else:
                        return  x_RA, y

HOME = os.environ['HOME']

def resize_imgs(x, size):
    ''' 目前只能处理单通道 
        x [n, 28, 28]
        size int
    '''
    resize_x = np.zeros([x.shape[0], size, size])
    for i, im in enumerate(x):
        im = Image.fromarray(im)
        im = im.resize([size, size], Image.ANTIALIAS)
        resize_x[i] = np.asarray(im)
    return resize_x

def load_mnist(split='train', translate=None, twox=False, ntr=None, autoaug=None, factor_num=16, randm=False,randn=False,channels=3,n=3,stride=5):
    '''
        autoaug == 'AA', AutoAugment
                   'FastAA', Fast AutoAugment
                   'RA', RandAugment
        channels == 3 默认返回 rgb 3通道图像
                    1 返回单通道图像
    '''
    path = f'data/mnist-{split}.pkl'
    if not os.path.exists(path):
        dataset = MNIST(f'{HOME}/.pytorch/MNIST', train=(split=='train'), download=True)
        x, y = dataset.data, dataset.targets
        if split=='train':
            x, y = x[0:10000], y[0:10000]
        x = torch.tensor(resize_imgs(x.numpy(), 32))
        x = (x.float()/255.).unsqueeze(1).repeat(1,3,1,1)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        # print("reading!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # 如果没有数据增强
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    
    # 数据增强管道
    transform = [transforms.ToPILImage()]
    transform_single_factor = [transforms.ToPILImage()]
    if autoaug == 'CA' or autoaug == 'CA_multiple':
        transform_CA = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
        transform_single_factor.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug == 'CA' or autoaug == 'CA_multiple':
            transform_CA.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'CA':
            print("--------------------------CA--------------------------")
            print("n:",n)
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(CounterfactualAugment_incausal(factor_num))
        elif autoaug == 'CA_multiple':
            print("--------------------------CA_multiple--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'Ours_A':
            print("--------------------------Ours_Augment--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))

    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    transform_single_factor.append(transforms.ToTensor())
    transform_single_factor = transforms.Compose(transform_single_factor)
    if autoaug == 'CA' or autoaug == 'CA_multiple':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, transform3=transform_CA,twox=twox)
    else:
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, twox=twox)
    # print(x.shape)
    # print(y.shape)
    return dataset

def load_cifar10(split='train', translate=None, twox=False, autoaug=None, factor_num=16, randm=False,randn=False,channels=3,n=3,stride=5):
    dataset = CIFAR10(f'{HOME}/.pytorch/CIFAR10', train=(split=='train'), download=True)
    x, y = dataset.data, dataset.targets
    x = x.transpose(0,3,1,2)
    x, y = torch.tensor(x), torch.tensor(y)
    x = x.float()/255.
    print(x.shape,y.shape)
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    #x.transpose(0,3,1,2)
    
    # 数据增强管道
    transform = [transforms.ToPILImage()]
    transform_single_factor = [transforms.ToPILImage()]
    if autoaug == 'CA' or autoaug == 'CA_multiple':
        transform_CA = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
        transform_single_factor.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug == 'CA' or autoaug == 'CA_multiple':
            transform_CA.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'CA':
            print("--------------------------CA--------------------------")
            print("n:",n)
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(CounterfactualAugment_incausal(factor_num))
        elif autoaug == 'CA_multiple':
            print("--------------------------CA_multiple--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'Ours_A':
            print("--------------------------Ours_Augment--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))

    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    transform_single_factor.append(transforms.ToTensor())
    transform_single_factor = transforms.Compose(transform_single_factor)
    if autoaug == 'CA' or autoaug == 'CA_multiple':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, transform3=transform_CA,twox=twox)
    else:
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, twox=twox)
    # print(x.shape)
    # print(y.shape)
    return dataset
def load_IMG(task='S-U', translate=None, twox=False, autoaug=None, factor_num=16, randm=False,randn=False,channels=3,n=3,stride=5):
    # path = f'data/img2vid/{domain}/stanford40_12.npz'
    if task == 'S-U':
        path = f'data/img2vid/{task}/stanford40_12.npz'
    elif task == 'E-H':
        path = f'data/img2vid/{task}/EAD50_13.npz'
    print(path)
    dataset = np.load(path)
    x, y = dataset['x'], dataset['y']
    b, g, r = np.split(x,3,axis=-1)
    x = np.concatenate((r,g,b),axis=-1)
    x = x.transpose(0,3,1,2)
    x, y = torch.tensor(x), torch.tensor(y, dtype=torch.long)
    x = x.float()/255.
    print(path,x.shape,y.shape)
    # for i in range(20):
    #     img_temp = transforms.ToPILImage()(x[i])
    #     img_temp.save('data/PACS/debug_images/img_pil_'+domain+'_'+split+'_'+str(i)+'.png')    
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    #x.transpose(0,3,1,2)
    
    # 数据增强管道
    transform = [transforms.ToPILImage()]
    if autoaug != 'CA_multiple_noSingle':
        transform_single_factor = [transforms.ToPILImage()]
    if autoaug == 'CA' or autoaug == 'CA_multiple' or autoaug == 'CA_multiple_noSingle':
        transform_CA = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug != 'CA_multiple_noSingle':
            transform_single_factor.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug == 'CA' or autoaug == 'CA_multiple' or autoaug == 'CA_multiple_noSingle':
            transform_CA.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'CA':
            print("--------------------------CA--------------------------")
            print("n:",n)
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(CounterfactualAugment_incausal(factor_num))
        elif autoaug == 'CA_multiple':
            print("--------------------------CA_multiple--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'CA_multiple_noSingle':
            print("--------------------------CA_multiple_noSingle--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            # transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'Ours_A':
            print("--------------------------Ours_Augment--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))

    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    if autoaug != 'CA_multiple_noSingle':
        transform_single_factor.append(transforms.ToTensor())
        transform_single_factor = transforms.Compose(transform_single_factor)
    if autoaug == 'CA' or autoaug == 'CA_multiple':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, transform3=transform_CA,twox=twox)
    elif autoaug == 'CA_multiple_noSingle':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform3=transform_CA,twox=twox)
    else:
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, twox=twox)
    # print(x.shape)
    # print(y.shape)
    return dataset

def load_VID(task='S-U',split='1'):
    if task == 'S-U':
        path = f'data/img2vid/{task}/ucf101_12_frame_sample8_{split}.npz'
    elif task == 'E-H':
        path = f'data/img2vid/{task}/hmdb51_13_frame_sample8_{split}.npz'
    dataset = np.load(path)
    print(path)
    x, y = dataset['x'], dataset['y']
    b, g, r = np.split(x,3,axis=-1)
    x = np.concatenate((r,g,b),axis=-1)
    x = x.transpose(0,3,1,2)
    x, y = torch.tensor(x), torch.tensor(y, dtype=torch.long)
    x = x.float()/255.
    print(path,x.shape,y.shape)
    # for i in range(20):
    #     img_temp = transforms.ToPILImage()(x[i])
    #     img_temp.save('data/PACS/debug_images/img_pil_'+domain+'_'+split+'_'+str(i)+'.png')    
    dataset = TensorDataset(x, y)
    return dataset

def load_pacs(domain='photo', split='train', translate=None, twox=False, autoaug=None, factor_num=16, randm=False,randn=False,channels=3,n=3,stride=5):
    path = f'data/PACS/{domain}_{split}.hdf5'
    dataset = h5py.File(path, 'r')
    x, y = dataset['images'], dataset['labels']
    for i in range(20):
        cv2.imwrite('data/PACS/debug_images/img_cv2_'+domain+'_'+split+'_'+str(i)+'.png', x[i])
    b, g, r = np.split(x,3,axis=-1)
    x = np.concatenate((r,g,b),axis=-1)
    x = x.transpose(0,3,1,2)
    x, y = torch.tensor(x), torch.tensor(y, dtype=torch.long)
    y = y - 1
    x = x.float()/255.
    print(path,x.shape,y.shape)
    # for i in range(20):
    #     img_temp = transforms.ToPILImage()(x[i])
    #     img_temp.save('data/PACS/debug_images/img_pil_'+domain+'_'+split+'_'+str(i)+'.png')    
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    #x.transpose(0,3,1,2)
    
    # 数据增强管道
    transform = [transforms.ToPILImage()]
    if autoaug != 'CA_multiple_noSingle':
        transform_single_factor = [transforms.ToPILImage()]
    if autoaug == 'CA' or autoaug == 'CA_multiple' or autoaug == 'CA_multiple_noSingle':
        transform_CA = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug != 'CA_multiple_noSingle':
            transform_single_factor.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug == 'CA' or autoaug == 'CA_multiple' or autoaug == 'CA_multiple_noSingle':
            transform_CA.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'CA':
            print("--------------------------CA--------------------------")
            print("n:",n)
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(CounterfactualAugment_incausal(factor_num))
        elif autoaug == 'CA_multiple':
            print("--------------------------CA_multiple--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'CA_multiple_noSingle':
            print("--------------------------CA_multiple_noSingle--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            # transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'Ours_A':
            print("--------------------------Ours_Augment--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))

    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    if autoaug != 'CA_multiple_noSingle':
        transform_single_factor.append(transforms.ToTensor())
        transform_single_factor = transforms.Compose(transform_single_factor)
    if autoaug == 'CA' or autoaug == 'CA_multiple':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, transform3=transform_CA,twox=twox)
    elif autoaug == 'CA_multiple_noSingle':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform3=transform_CA,twox=twox)
    else:
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, twox=twox)
    # print(x.shape)
    # print(y.shape)
    return dataset

def read_dataset(domain, split):
    path = f'data/PACS/{domain}_{split}.hdf5'
    dataset = h5py.File(path, 'r')
    x_temp, y_temp = dataset['images'], dataset['labels']
    b, g, r = np.split(x_temp,3,axis=-1)
    x_temp = np.concatenate((r,g,b),axis=-1)
    x_temp = x_temp.transpose(0,3,1,2)
    x_temp, y_temp = torch.tensor(x_temp), torch.tensor(y_temp, dtype=torch.long)
    y_temp = y_temp - 1
    x_temp = x_temp.float()/255.
    return x_temp, y_temp

def load_pacs_multi(target_domain=['photo'], split='train', translate=None, twox=False, autoaug=None, factor_num=16, randm=False,randn=False,channels=3,n=3,stride=5):
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    source_domain = [i for i in domains if i != target_domain]
    for i in range(len(source_domain)):
        x_temp, y_temp = read_dataset(source_domain[i],split=split)
        print(x_temp.shape,y_temp.shape)
        if i == 0:
            x = x_temp.clone()
            y = y_temp.clone()
        else:
            x = torch.cat([x,x_temp],0)
            y = torch.cat([y,y_temp],0)
    print(x.shape,y.shape)
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    #x.transpose(0,3,1,2)
    
    # 数据增强管道
    transform = [transforms.ToPILImage()]
    if autoaug != 'CA_multiple_noSingle':
        transform_single_factor = [transforms.ToPILImage()]
    if autoaug == 'CA' or autoaug == 'CA_multiple' or autoaug == 'CA_multiple_noSingle':
        transform_CA = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug != 'CA_multiple_noSingle':
            transform_single_factor.append(transforms.RandomAffine(0, [translate, translate]))
        if autoaug == 'CA' or autoaug == 'CA_multiple' or autoaug == 'CA_multiple_noSingle':
            transform_CA.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'CA':
            print("--------------------------CA--------------------------")
            print("n:",n)
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(CounterfactualAugment_incausal(factor_num))
        elif autoaug == 'CA_multiple':
            print("--------------------------CA_multiple--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'CA_multiple_noSingle':
            print("--------------------------CA_multiple_noSingle--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            # transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))
            transform_CA.append(MultiCounterfactualAugment_incausal(factor_num, stride))
        elif autoaug == 'Ours_A':
            print("--------------------------Ours_Augment--------------------------")
            transform.append(RandAugment_incausal(n,4,factor_num, randm=randm,randn=randn))
            transform_single_factor.append(FactualAugment_incausal(4, factor_num, randm=False))

    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    if autoaug != 'CA_multiple_noSingle':
        transform_single_factor.append(transforms.ToTensor())
        transform_single_factor = transforms.Compose(transform_single_factor)
    if autoaug == 'CA' or autoaug == 'CA_multiple':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, transform3=transform_CA,twox=twox)
    elif autoaug == 'CA_multiple_noSingle':
        transform_CA.append(transforms.ToTensor())
        transform_CA = transforms.Compose(transform_CA)
        dataset = myTensorDataset(x, y, transform=transform, transform3=transform_CA,twox=twox)
    else:
        dataset = myTensorDataset(x, y, transform=transform, transform2=transform_single_factor, twox=twox)
    # print(x.shape)
    # print(y.shape)
    return dataset


def load_cifar10_c_level1(dataroot):
    path = f'data/cifar10_c_level1.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level1")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0 
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[0:10000]
                index = index + 1
        y = torch.tensor(y)                              
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level1")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)        
    return dataset
def load_cifar10_c_level2(dataroot):
    path = f'data/cifar10_c_level2.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level2")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0 
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[10000:20000]
                index = index + 1
        y = torch.tensor(y)                              
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level2")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)        
    return dataset
def load_cifar10_c_level3(dataroot):
    path = f'data/cifar10_c_level3.pkl'
    if not os.path.exists(path):
        print("generating cifar10_c_level3")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0 
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[20000:30000]
                index = index + 1
        y = torch.tensor(y)                              
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level3")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)        
    return dataset
def load_cifar10_c_level4(dataroot):
    path = f'data/cifar10_c_level4.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level4")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0 
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[30000:40000]
                index = index + 1
        y = torch.tensor(y)                              
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level4")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)        
    return dataset
def load_cifar10_c_level5(dataroot):
    path = f'data/cifar10_c_level5.pkl'
    if not os.path.exists(path):
        print("genenrating cifar10_c_level5")
        labels = np.load(os.path.join(dataroot, 'labels.npy'))
        y_single = labels[0:10000]
        x = torch.zeros((190000,3,32,32))
        for j in range(19):
            if j == 0:
                y = y_single
            else:
                y = np.hstack((y,y_single))
        index = 0 
        for filename in os.listdir(dataroot):
            if filename=='labels.npy':
                continue
            else:
                imgs = np.load(os.path.join(dataroot,filename))
                imgs = imgs.transpose(0,3,1,2)
                imgs = torch.tensor(imgs)
                imgs = imgs.float()/255.
                print(imgs.shape)
                x[index*10000:(index+1)*10000] = imgs[40000:50000]
                index = index + 1
        y = torch.tensor(y)                              
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    else:
        print("reading cifar10_c_level5")
        with open(path, 'rb') as f:
            x, y = pickle.load(f)
    dataset = TensorDataset(x, y)        
    return dataset
def load_cifar10_c(dataroot):
    y = np.load(os.path.join(dataroot, 'labels.npy'))
    print("y.shape:",y.shape)
    y_single = y[0:10000]
    x1 = torch.zeros((190000,3,32,32))
    x2 = torch.zeros((190000,3,32,32))
    x3 = torch.zeros((190000,3,32,32))
    x4 = torch.zeros((190000,3,32,32))
    x5 = torch.zeros((190000,3,32,32))
    for j in range(19):
        if j == 0:
            y_total = y_single
        else:
            y_total = np.hstack((y_total,y_single))
    print("y_total.shape:",y_total.shape)
    index = 0 
    for filename in os.listdir(dataroot):
        if filename=='labels.npy':
            continue
        else:
            x = np.load(os.path.join(dataroot,filename))
            x = x.transpose(0,3,1,2)
            x = torch.tensor(x)
            x = x.float()/255.
            print(x.shape)
            x1[index*10000:(index+1)*10000] = x[0:10000]
            x2[index*10000:(index+1)*10000] = x[10000:20000]
            x3[index*10000:(index+1)*10000] = x[20000:30000]
            x4[index*10000:(index+1)*10000] = x[30000:40000]
            x5[index*10000:(index+1)*10000] = x[40000:50000]
            index = index + 1
    # x1, x2, x3, x4, x5, y_total = torch.tensor(x1), torch.tensor(x2), torch.tensor(x3),\
                                    # torch.tensor(x4),torch.tensor(x5),torch.tensor(y_total)
    y_total = torch.tensor(y_total)                              
    dataset1 = TensorDataset(x1, y_total)
    dataset2 = TensorDataset(x2, y_total)
    dataset3 = TensorDataset(x3, y_total)
    dataset4 = TensorDataset(x4, y_total)
    dataset5 = TensorDataset(x5, y_total)
    return dataset1,dataset2,dataset3,dataset4,dataset5

def load_cifar10_c_class(dataroot,CORRUPTIONS):
    y = np.load(os.path.join(dataroot, 'labels.npy'))
    y_single = y[0:10000]
    y_single = torch.tensor(y_single) 
    print("y.shape:",y.shape)
    x = np.load(os.path.join(dataroot,CORRUPTIONS+'.npy'))
    print("loading data of",os.path.join(dataroot,CORRUPTIONS+'.npy'))
    x = x.transpose(0,3,1,2)
    x = torch.tensor(x)
    x = x.float()/255.
    dataset = []
    for i in range(5):
        x_single = x[i*10000:(i+1)*10000]
        dataset.append(TensorDataset(x_single, y_single))
    return dataset

def load_usps(split='train', channels=3):
    path = f'data/usps-{split}.pkl'
    if not os.path.exists(path):
        dataset = USPS(f'{HOME}/.pytorch/USPS', train=(split=='train'), download=True)
        x, y = dataset.data, dataset.targets
        x = torch.tensor(resize_imgs(x, 32))
        x = (x.float()/255.).unsqueeze(1).repeat(1,3,1,1)
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    dataset = TensorDataset(x, y)
    return dataset

def load_svhn(split='train', channels=3):
    dataset = SVHN(f'{HOME}/.pytorch/SVHN', split=split, download=True)
    x, y = dataset.data, dataset.labels
    x = x.astype('float32')/255.
    x, y = torch.tensor(x), torch.tensor(y)
    if channels == 1:
        x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset


def load_syndigit(split='train', channels=3):
    path = f'data/synth_{split}_32x32.mat'
    data = loadmat(path)
    x, y = data['X'], data['y']
    x = np.transpose(x, [3, 2, 0, 1]).astype('float32')/255.
    y = y.squeeze()
    x, y = torch.tensor(x), torch.tensor(y)
    if channels == 1:
        x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

def load_mnist_m(split='train', channels=3):
    path = f'data/mnist_m-{split}.pkl'
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        x, y = torch.tensor(x.astype('float32')/255.), torch.tensor(y)
        if channels==1:
            x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

if __name__=='__main__':
    dataset = load_mnist(split='train')
    print('mnist train', len(dataset))
    dataset = load_mnist('test')
    print('mnist test', len(dataset))
    dataset = load_mnist_m('test')
    print('mnsit_m test', len(dataset))
    dataset = load_svhn(split='test')
    print('svhn', len(dataset))
    dataset = load_usps(split='test')
    print('usps', len(dataset))
    dataset = load_syndigit(split='test')
    print('syndigit', len(dataset))

