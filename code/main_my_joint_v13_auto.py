
'''
训练 base 模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast,GradScaler

import os
import click
import time
import numpy as np

from network import mnist_net_my as mnist_net
from network import wideresnet as wideresnet
from network import resnet as resnet
from network import adaptor_v2

from tools import causalaugment_v3 as causalaugment
import data_loader_joint_v3 as data_loader
# from utils import set_requires_grad

HOME = os.environ['HOME']

@click.command()
@click.option('--gpu', type=str, default='0', help='选择gpu')
@click.option('--data', type=str, default='mnist', help='数据集名称')
@click.option('--ntr', type=int, default=None, help='选择训练集前ntr个样本')
@click.option('--translate', type=float, default=None, help='随机平移数据增强')
@click.option('--autoaug', type=str, default=None, help='AA FastAA RA')
@click.option('--n', type=int, default=3, help='选择多少个factor生成RA')
@click.option('--stride', type=int, default=5, help='if autoaug==CA_multiple, stride is used')
@click.option('--factor_num', type=int, default=16, help='the first n factors')
@click.option('--epochs', type=int, default=100)
@click.option('--nbatch', type=int, default=100, help='每个epoch中batch的数量')
@click.option('--batchsize', type=int, default=128, help='每个batch中样本的数量')
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='是否选择学习率衰减策略')
@click.option('--svroot', type=str, default='./saved', help='项目文件保存路径')
@click.option('--clsadapt', type=bool, default=True, help='映射后是否用分类损失')
@click.option('--lambda_causal', type=float, default=1, help='the weight of reconstruction during mapping and causal ')
@click.option('--lambda_re', type=float, default=1, help='the weight of reconstruction during mapping and causal ')
@click.option('--randm', type=bool, default=True, help='m取值是否randm')
@click.option('--randn', type=bool, default=False, help='原始特征是否detach')
@click.option('--network', type=str, default='resnet18', help='项目文件保存路径')
def experiment(gpu, data, ntr, translate, autoaug,n,stride, factor_num, epochs, nbatch, batchsize, lr, lr_scheduler, svroot, clsadapt, lambda_causal,lambda_re,randm,randn,network):
    
    settings = locals().copy()
    print(settings)

    # 全局设置
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if not os.path.exists(svroot):
        os.makedirs(svroot)
    log_file = open(svroot+os.sep+'log.log',"w")
    log_file.write(str(settings)+'\n')
    writer = SummaryWriter(svroot)
    CA = causalaugment.MultiCounterfactualAugment(factor_num,stride)   
    # FA = causalaugment.FactualAugment(m=4, factor_num=factor_num, randm=True)
    # 加载数据集和模型
    if data in ['mnist', 'mnist_t']: 
        if data == 'mnist':
            trset = data_loader.load_mnist('train', translate=translate,twox=True, ntr=ntr, factor_num=factor_num,autoaug=autoaug,randm=randm,randn=randn,n=n,stride=stride)
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', translate=translate, ntr=ntr)
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=0, \
                sampler=RandomSampler(trset, True, nbatch*batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=0, shuffle=False)
        cls_net = mnist_net.ConvNet().cuda()
        AdaptNet = []
        parameter_list = []
        for i in range(factor_num):
            mapping = adaptor_v2.mapping(1024,512,1024,2).cuda()
            AdaptNet.append(mapping)
            parameter_list.append({'params':mapping.parameters(),'lr':lr})
        if autoaug == 'CA_multiple':
            var_num = len(list(range(0, 31, stride)))
            E_to_W = adaptor_v2.effect_to_weight(10,100,1).cuda()
        else:
            E_to_W = adaptor_v2.effect_to_weight(10,100,1).cuda()
        parameter_list.append({'params':cls_net.parameters(),'lr':lr})
        parameter_list.append({'params':E_to_W.parameters(),'lr':lr})
        #print("---------------------------------------------------------------------------------------")
        opt = optim.Adam(parameter_list, lr=lr)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        elif lr_scheduler == 'Exp':
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.95) 
        elif lr_scheduler == 'Step':
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=int(epochs*0.8))
        # print("------------------------------------opt_mapping---------------------------------------------------")
        # for param_group in opt_mapping.param_groups:
        #     print(param_group.keys())
        #     # print(type(param_group))
        #     print([type(value) for value in param_group.values()])
        #     print('lr: ',param_group['lr'])

        # print("------------------------------------opt_causal---------------------------------------------------")
        # for param_group in opt_causal.param_groups:
        #     print(param_group.keys())
        #     # print(type(param_group))
        #     print([type(value) for value in param_group.values()])
        #     print('lr: ',param_group['lr'])
    
    elif data == 'cifar10':
        # 加载数据集
        trset = data_loader.load_cifar10(split='train',twox=True, factor_num=factor_num,autoaug=autoaug,randm=randm,randn=randn,n=n,stride=stride)
        teset = data_loader.load_cifar10(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=4, shuffle=False)
        cls_net = wideresnet.WideResNet(16, 10, 4).cuda()
        # cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        AdaptNet = []
        parameter_list = []
        for i in range(factor_num):
            mapping = adaptor_v2.mapping(256,512,256,4).cuda()
            AdaptNet.append(mapping)
            parameter_list.append({'params':mapping.parameters(),'lr':lr})
        if autoaug == 'CA_multiple':
            var_num = len(list(range(0, 31, stride)))
            E_to_W = adaptor_v2.effect_to_weight(10,100,1).cuda()
        else:
            E_to_W = adaptor_v2.effect_to_weight(10,100,1).cuda()
        parameter_list.append({'params':cls_net.parameters(),'lr':lr})
        parameter_list.append({'params':E_to_W.parameters(),'lr':lr})
        #print("---------------------------------------------------------------------------------------")
        # opt = optim.Adam(parameter_list)
        opt = optim.SGD(parameter_list, lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        elif lr_scheduler == 'Exp':
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
        elif lr_scheduler == 'Step':
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=int(epochs*0.8))
    elif data in ['art_painting', 'cartoon', 'photo', 'sketch']:
        # 加载数据集
        trset = data_loader.load_pacs(domain=data, split='train', twox=True, factor_num=factor_num,autoaug=autoaug,randm=randm,randn=randn,n=n,stride=stride)
        teset = data_loader.load_pacs(domain=data, split='val')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=4, shuffle=False)
        if network == 'resnet18':
            cls_net = resnet.resnet18(classes=7,c_dim=2048).cuda()
            input_dim = 2048
            # for param in cls_net.features.parameters():
            #     param.requires_grad = False
            # for name, parms in cls_net.named_parameters():  
            #     print('-->name:', name)
            #     print('-->grad_requirs:',parms.requires_grad)
        # cls_opt = optim.SGD(cls_net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        # print(cls_net.state_dict())

        classifier_param = list(map(id, cls_net.class_classifier.parameters()))
        backbone_param  =  filter(lambda p: id(p) not in classifier_param and p.requires_grad, cls_net.parameters())
        AdaptNet = []
        parameter_list = []
        for i in range(factor_num):
            mapping = adaptor_v2.mapping(input_dim,1024,input_dim,4).cuda()
            AdaptNet.append(mapping)
            parameter_list.append({'params':mapping.parameters(),'lr':lr})
        if autoaug == 'CA_multiple':
            var_num = len(list(range(0, 31, stride)))
            E_to_W = adaptor_v2.effect_to_weight(7,70,1).cuda()
        else:
            E_to_W = adaptor_v2.effect_to_weight(7,70,1).cuda()
        parameter_list.append({'params':backbone_param,'lr':0.01*lr})
        parameter_list.append({'params':cls_net.class_classifier.parameters(),'lr':lr})
        parameter_list.append({'params':E_to_W.parameters(),'lr':lr})
        #print("---------------------------------------------------------------------------------------")
        # opt = optim.Adam(parameter_list)

        opt = optim.SGD(parameter_list, momentum=0.9, nesterov=True, weight_decay=5e-4)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        elif lr_scheduler == 'Exp':
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.99999) 
        elif lr_scheduler == 'Step':
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=15)
    elif 'synthia' in data:
        # 加载数据集
        branch = data.split('_')[1]
        trset = data_loader.load_synthia(branch)
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        teloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        imsize = [192, 320]
        nclass = 14
        # 加载模型
        cls_net = fcn.FCN_resnet50(nclass=nclass).cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)#, weight_decay=1e-4) # 对于synthia 加上weigh_decay会掉1-2个点
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs*len(trloader))
    
    cls_criterion = nn.CrossEntropyLoss()
    adapt_criterion = nn.MSELoss()
    # 开始训练
    best_acc = 0
    best_acc_t = 0
    scaler = GradScaler()
    for epoch in range(epochs):
        t1 = time.time() 
        loss_list = []
        cls_net.train()
        # unloader = transforms.ToPILImage()
        print(len(trloader))
        for i, (x_four,y) in enumerate(trloader):
            b_sample_num = y.size(0)
            x, x_RA, x_FA, x_CA, y = x_four[0].cuda(), x_four[1].cuda(), x_four[2].cuda(), x_four[3].cuda(), y.cuda()
            b, c, h, w = x.shape
            # x_FA_ = x_FA.transpose(1,2)
            x_FA = x_FA.reshape(b*factor_num, c, h, w)
            x_CA = x_CA.reshape(b*factor_num*var_num, c, h, w)
            #learning mapping
            y_repeat = y.unsqueeze(0).reshape(b_sample_num,1).repeat((1,factor_num)).reshape(1,b_sample_num*factor_num).squeeze()
            # x_FA = FA(x).cuda().detach()
            # x_CA = CA(x_RA).cuda().detach()
            with autocast():
                p,f = cls_net(x)
                # print("x.shape:",x.shape)
                # print("x_FA.shape:",x_FA.shape)
                _,f_FA = cls_net(x_FA)
                p_RA,f_RA = cls_net(x_RA)
                p_CA,_ = cls_net(x_CA)
                # print("f.shape:",f.shape)
                # print("f_FA.shape:",f_FA.shape)
                #learning mapping
                f_repeat = f.repeat((1,factor_num)).reshape(f_FA.shape)
                f_adapt = torch.zeros(f_FA.shape).cuda()
                for b in range(b_sample_num):
                    for j in range(factor_num):
                        f_adapt[b*factor_num+j] = AdaptNet[j](f_FA[b*factor_num+j])
                p_adapt = cls_net(f_adapt, mode='c')

                #learning causality
                if autoaug == 'CA_multiple':
                    p_RA_repeat = p_RA.repeat((1,factor_num*var_num)).reshape(p_CA.shape)
                    effect_context = p_RA_repeat - p_CA
                    effect_context = effect_context.reshape(b_sample_num,factor_num,var_num,-1)
                    effect_context = effect_context.mean(axis=2).reshape(b_sample_num*factor_num,-1)
                    # print("effect_context.shape:",effect_context.shape)
                else:
                    p_RA_repeat = p_RA.repeat((1,factor_num)).reshape(p_CA.shape)
                    effect_context = p_RA_repeat - p_CA
                weight = E_to_W(effect_context)
                # weight = E_to_W(effect_context.detach())
                weight = weight.reshape(b_sample_num,factor_num)
                alphas = F.softmax(weight,dim=1)
                
                f_adapt_RA = torch.zeros(f_RA.shape).cuda()
                for b in range(b_sample_num):
                    for j in range(factor_num):
                        f_adapt_RA[b] = f_adapt_RA[b]+ alphas[b,j]*AdaptNet[j](f_RA[b])     
                p_adapt_RA = cls_net(f_adapt_RA, mode='c')
                
                cls_loss = cls_criterion(p, y)
                re_mapping = adapt_criterion(f_adapt,f_repeat) 
                re_causal = adapt_criterion(f_adapt_RA,f)                
                cls_loss_mapping = cls_criterion(p_adapt, y_repeat)
                cls_loss_causal = cls_criterion(p_adapt_RA, y)

                loss = cls_loss + cls_loss_mapping + lambda_re*re_mapping + lambda_causal*(lambda_re*re_causal + cls_loss_causal)

            opt.zero_grad()            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_list.append([cls_loss.item(), cls_loss_mapping.item(),cls_loss_causal.item(), re_mapping.item(), re_causal.item()])
            
            # 调整学习率
        if lr_scheduler in ['cosine', 'Exp', 'Step']:
            writer.add_scalar('scalar/lr', opt.param_groups[0]["lr"], epoch)
            print(opt.param_groups[0]["lr"])
            print("changing lr")
            scheduler.step()
        cls_loss, cls_loss_mapping, cls_loss_causal, re_mapping, re_causal = np.mean(loss_list, 0)    

        # 测试，并保存最优模型
        cls_net.eval()
        if data in ['mnist', 'mnist_t', 'cifar10', 'mnistvis', 'art_painting', 'cartoon', 'photo', 'sketch']:
            teacc = evaluate(cls_net, teloader)

        elif 'synthia' in data:
            teacc = evaluate_seg(cls_net, teloader, nclass) # 这里算的其实是 miou

        if best_acc < teacc:
            print(f'---------------------saving model at epoch {epoch}----------------------------------------------------')
            log_file.write(f'saving model at epoch {epoch}\n')

            best_acc = teacc
            torch.save(cls_net.state_dict(),os.path.join(svroot, 'best_cls_net.pkl'))
            for j in range(factor_num):
                torch.save(AdaptNet[j].state_dict(),os.path.join(svroot, 'best_mapping_'+str(j)+'.pkl'))
            torch.save(E_to_W.state_dict(), os.path.join(svroot, 'best_E_to_W.pkl'))

        # 保存日志
        t2 = time.time()
        print(f'epoch {epoch}, time {t2-t1:.2f}, cls_loss {cls_loss:.4f} cls_loss_mapping {cls_loss_mapping:.4f} cls_loss_causal {cls_loss_causal:.4f} re_mapping {re_mapping:.4f} re_causal {re_causal:.4f} /// teacc {teacc:2.2f} lr {opt.param_groups[0]["lr"]:.8f}')
        log_file.write(f'epoch {epoch}, time {t2-t1:.2f}, cls_loss {cls_loss:.4f} cls_loss_mapping {cls_loss_mapping:.4f} cls_loss_causal {cls_loss_causal:.4f} re_mapping {re_mapping:.4f} re_causal {re_causal:.4f} /// teacc {teacc:2.2f} lr {opt.param_groups[0]["lr"]:.8f} \n')
        writer.add_scalar('scalar/cls_loss', cls_loss, epoch)
        writer.add_scalar('scalar/cls_loss_mapping', cls_loss_mapping, epoch)
        writer.add_scalar('scalar/cls_loss_causal', cls_loss_causal, epoch)
        writer.add_scalar('scalar/re_mapping', re_mapping, epoch)
        writer.add_scalar('scalar/re_causal', re_causal, epoch)
        writer.add_scalar('scalar/teacc', teacc, epoch)
    print(f'---------------------saving last model at epoch {epoch}----------------------------------------------------')
    log_file.write(f'saving last model at epoch {epoch}\n')
    torch.save(cls_net.state_dict(),os.path.join(svroot, 'last_cls_net.pkl'))
    for j in range(factor_num):
        torch.save(AdaptNet[j].state_dict(),os.path.join(svroot, 'last_mapping_'+str(j)+'.pkl'))
    torch.save(E_to_W.state_dict(), os.path.join(svroot, 'last_E_to_W.pkl'))

    writer.close()
def evalute_pacs(source_domain,cls_net,CA,AdaptNet,E_to_W):
    cls_net.eval()
    data_total = ['art_painting', 'cartoon', 'photo', 'sketch']
    target = [i for i in data_total if i!=source_domain]
    acc_CA = np.zeros(len(target))
    for idx, data in enumerate(target):
        teset = data_loader.load_pacs(data, 'test')
        teloader = DataLoader(teset, batch_size=6, num_workers=0)
        # 计算评价指标
        acc_CA[idx] = evaluate_causal(cls_net, teloader, CA, AdaptNet, E_to_W)
    acc_avg_CA = sum(acc_CA)/len(target)
    return acc_avg_CA,acc_CA


def evaluate(net, teloader):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        with torch.no_grad():
            x1 = x1.cuda()
            p1,_ = net(x1, mode='fc')
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc
def extract_feature(net, teloader, savedir):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        img_class = y1[0].cpu().numpy()
        save_path = os.path.join(savedir,str(img_class))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with torch.no_grad():
            x1 = x1.cuda()
            p1,f1 = net(x1, mode='fc')
            save_name = save_path+os.sep+str(i)+'.npy'
            np.save(save_name,f1.cpu())
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc

def evaluate_causal(net, teloader, CA, AdaptNet, E_to_W):
    ps = []
    ys = []
    p_orig = []
    y_orig = []
    for i,(x1, y1) in enumerate(teloader):
        b_sample_num = x1.size(0)
        with torch.no_grad():
            x1 = x1.cuda()
            p1,f_x1 = net(x1, mode='fc')
            x1_CA = CA(x1).cuda()
            p1_CA,_ = net(x1_CA, mode='fc')
            p1_repeat = p1.repeat((1,CA.factor_num*CA.var_num)).reshape(p1_CA.shape)
            effect_context = p1_repeat - p1_CA
            effect_context = effect_context.reshape(b_sample_num,CA.factor_num,CA.var_num,-1)
            effect_context = effect_context.mean(axis=2).reshape(b_sample_num*CA.factor_num,-1)
            weight = E_to_W(effect_context)
            weight = weight.reshape(b_sample_num,CA.factor_num)
            alphas = F.softmax(weight,dim=1)
            f_adapt = torch.zeros(f_x1.shape).cuda()
            for b in range(b_sample_num):
                for j in range(CA.factor_num):
                    f_adapt[b] = f_adapt[b]+ alphas[b,j]*AdaptNet[j](f_x1[b])
            p_adapt = net(f_adapt, mode='c')
            p_adapt = p_adapt.argmax(dim=1)
            ps.append(p_adapt.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc

def extract_feature_do(net, teloader, CA, AdaptNet, E_to_W, savedir_base, savedir,source_flag):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        img_class = y1[0].cpu().numpy()
        save_path_base = os.path.join(savedir_base,str(img_class))
        save_path = os.path.join(savedir,str(img_class))
        if not os.path.exists(save_path_base):
            os.makedirs(save_path_base)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        b_sample_num = x1.size(0)
        with torch.no_grad():
            x1 = x1.cuda()
            p1,f_x1 = net(x1, mode='fc')
            save_name_base = save_path_base+os.sep+str(i)+'_base.npy'
            print(save_name_base)
            np.save(save_name_base,f_x1.cpu())            
            x1_CA = CA(x1).cuda()
            p1_CA,_ = net(x1_CA, mode='fc')
            p1_repeat = p1.repeat((1,CA.factor_num*CA.var_num)).reshape(p1_CA.shape)
            effect_context = p1_repeat - p1_CA
            effect_context = effect_context.reshape(b_sample_num,CA.factor_num,CA.var_num,-1)
            effect_context = effect_context.mean(axis=2).reshape(b_sample_num*CA.factor_num,-1)
            weight = E_to_W(effect_context)
            weight = weight.reshape(b_sample_num,CA.factor_num)
            alphas = F.softmax(weight,dim=1)
            f_adapt = torch.zeros(f_x1.shape).cuda()
            for b in range(b_sample_num):
                for j in range(CA.factor_num):
                    f_adapt[b] = f_adapt[b]+ alphas[b,j]*AdaptNet[j](f_x1[b])
            if not source_flag:
                save_name = save_path+os.sep+str(i)+'.npy'
                print(save_name)
                np.save(save_name,f_adapt.cpu())
            p_adapt = net(f_adapt, mode='c')
            p_adapt = p_adapt.argmax(dim=1)
            ps.append(p_adapt.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc


def evaluate_mapping(net, teloader, FA, AdaptNet, source=False):
    correct, count = 0, 0
    ps = []
    ys = []
    pt = []
    yt = []
    factor_num = FA.factor_num
    for j in range(factor_num):
        ps.append([])
        ys.append([])
        pt.append([])
        yt.append([])
    ps.append([])
    ys.append([])   
    # print(len(ps),len(ys))
    for i,(x1, y1) in enumerate(teloader):
        with torch.no_grad():
            x1 = x1.cuda()
            b = x1.size(0)
            if source:
                x_FA = FA(x1).cuda()
                _, f = net(x_FA, mode='fc')
                p,_ = net(x1, mode='fc')
                p = p.argmax(dim=1)
                ps[-1].append(p.detach().cpu().numpy())
                ys[-1].append(y1.numpy())
            else:
                p, f = net(x1, mode='fc')
                f = f.repeat((1,factor_num)).reshape((-1,f.size(1)))         
                p = p.argmax(dim=1)
                ps[-1].append(p.detach().cpu().numpy())
                ys[-1].append(y1.numpy())
            for b_ in range(b):
                for j in range(factor_num):
                    f_adapt = AdaptNet[j](f[b_*factor_num+j])
                    #f_adapt = torch.mm(AdaptNet[j].W1,f_FA[b_*factor_num+j].unsqueeze(1)).squeeze()
                    p1 = net(f_adapt, mode='c')
                    p1 = p1.argmax(dim=0)
                    ps[j].append(p1.detach().cpu())
                    ys[j].append(y1[b_])
                    p1_t = net(f[b_*factor_num+j], mode='c')
                    # print("p1_t.shape:",p1_t.shape)
                    p1_t = p1_t.argmax(dim=0)
                    pt[j].append(p1_t.detach().cpu())
                    yt[j].append(y1[b_])
    # 计算评价指标
    acc = np.zeros(factor_num+1)
    acc_t = np.zeros(factor_num+1)
    for j in range(factor_num):
        pred = torch.stack(ps[j])
        label = torch.stack(ys[j])
        acc[j] = (pred==label).sum()/float(len(ys[j]))*100
        predt = torch.stack(pt[j])
        labelt = torch.stack(yt[j])
        acc_t[j] = (predt==labelt).sum()/float(len(yt[j]))*100
    pred = np.concatenate(ps[-1])
    label = np.concatenate(ys[-1])
    acc[-1] = np.mean(pred==label)*100
    # print("acc:",acc)
    return acc, acc_t
def evaluate_causal_with_entropy(net, teloader, CA, AdaptNet):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        b_sample_num = x1.size(0)
        with torch.no_grad():
            x1 = x1.cuda()
            p1,f_x1 = net(x1, mode='fc')
            
            x1_CA = CA(x1).cuda()
            p1_CA, _ = net(x1_CA, mode='fc')
            p1_repeat = p1.repeat((1,CA.factor_num*CA.var_num)).reshape(p1_CA.shape)
            effect_context = p1_repeat - p1_CA
            effect_context = effect_context.reshape(b_sample_num,CA.factor_num,CA.var_num,-1)
            effect_context = effect_context.mean(axis=2).reshape(b_sample_num*CA.factor_num,-1)
            effect_context = F.softmax(effect_context,dim=1)
            # weight = calc_ent(effect_context)
            weight = torch.sum(-effect_context*(torch.log2(effect_context)),dim=1)
            weight = weight.reshape(b_sample_num,CA.factor_num)
            alphas = F.softmax(-weight,dim=1)
            f_adapt = torch.zeros(f_x1.shape).cuda()
            for b in range(b_sample_num):
                for j in range(CA.factor_num):
                    f_adapt[b] = f_adapt[b]+ alphas[b,j]*AdaptNet[j](f_x1[b]) 
            p_adapt = net(f_adapt, mode='c')
            p_adapt = p_adapt.argmax(dim=1)
            ps.append(p_adapt.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc
def evaluate_causal_with_average(net, teloader, factor_num, AdaptNet):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(teloader):
        b_sample_num = x1.size(0)
        with torch.no_grad():
            x1 = x1.cuda()
            p1,f_x1 = net(x1, mode='fc')
            f_adapt = torch.zeros(f_x1.shape).cuda()
            for b in range(b_sample_num):
                for j in range(factor_num):
                    f_adapt[b] = f_adapt[b]+ float(1/factor_num)*AdaptNet[j](f_x1[b]) 
            p_adapt = net(f_adapt, mode='c')
            p_adapt = p_adapt.argmax(dim=1)
            ps.append(p_adapt.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    return acc
if __name__=='__main__':
    experiment()