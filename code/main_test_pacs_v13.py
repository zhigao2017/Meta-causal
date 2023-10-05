
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import numpy as np
import click
import pandas as pd

from network import resnet as resnet
from network import adaptor_v2
from tools import causalaugment_v3 as causalaugment
from main_my_joint_v13_auto import evaluate,evaluate_causal,evaluate_causal_with_entropy,evaluate_mapping,evaluate_causal_with_average
import data_loader_joint_v3 as data_loader

@click.command()
@click.option('--gpu', type=str, default='0', help='选择GPU编号')
@click.option('--svroot', type=str, default='./saved')
@click.option('--source_domain', type=str, default='art_painting', help='source domain')
@click.option('--svpath', type=str, default=None, help='保存日志的路径')
@click.option('--factor_num', type=int, default=16)
@click.option('--epoch', type=str, default='best')
@click.option('--stride', type=int, default=5)
@click.option('--eval_mapping', type=bool, default=False, help='是否查看mapping学习效果')
@click.option('--network', type=str, default='resnet18', help='项目文件保存路径')
def main(gpu, svroot, source_domain, svpath, factor_num, epoch, stride,eval_mapping, network):
    evaluate_pacs(gpu, svroot, source_domain, svpath, factor_num, epoch, stride,eval_mapping, network)
    
def evaluate_pacs(gpu, svroot, source_domain, svpath, factor_num=16, epoch='best', stride=5,eval_mapping=False, network='resnet18'):
    settings = locals().copy()
    print(settings)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # 加载分类模型
    if network == 'resnet18':
        cls_net = resnet.resnet18(classes=7,c_dim=2048).cuda()
        input_dim = 2048
    if epoch == 'best':
        print("loading weight of %s"%(epoch))
        saved_weight = torch.load(os.path.join(svroot, 'best_cls_net.pkl'))
    elif epoch == 'last':
        print("loading weight of %s"%(epoch))
        saved_weight = torch.load(os.path.join(svroot, 'last_cls_net.pkl'))
    cls_net.load_state_dict(saved_weight)
    cls_net.eval()
    # 加载adaptation模型
    FA = causalaugment.FactualAugment(m=4, factor_num=factor_num)
    CA = causalaugment.MultiCounterfactualAugment(factor_num,stride) 
    AdaptNet = []
    parameter_list = []
    for i in range(factor_num):
        if epoch == 'best':
            print("loading weight of %s"%(epoch))
            saved_weight = torch.load(os.path.join(svroot, 'best_mapping_'+str(i)+'.pkl'))
        elif epoch == 'last':
            print("loading weight of %s"%(epoch))
            saved_weight = torch.load(os.path.join(svroot, 'last_mapping_'+str(i)+'.pkl'))
        # saved_weight = torch.load(os.path.join(svroot, 'best_mapping_'+str(i)+'.pkl'))
        mapping = adaptor_v2.mapping(input_dim,1024,input_dim,4).cuda()
        mapping.load_state_dict(saved_weight)
        AdaptNet.append(mapping)
    if epoch == 'best':
        print("loading weight of %s"%(epoch))
        saved_weight = torch.load(os.path.join(svroot, 'best_E_to_W.pkl'))
    elif epoch == 'last':
        print("loading weight of %s"%(epoch))
        saved_weight = torch.load(os.path.join(svroot, 'last_E_to_W.pkl'))
    E_to_W = adaptor_v2.effect_to_weight(7,70,1).cuda()
    # Color_mapping.load_state_dict(saved_weight['Color_mapping'])
    # Contrast_mapping.load_state_dict(saved_weight['Contrast_mapping'])
    # Brightness_mapping.load_state_dict(saved_weight['Brightness_mapping'])
    # saved_weight = torch.load(os.path.join(svroot, 'best_E_to_W.pkl'))
    E_to_W.load_state_dict(saved_weight)

    # 测试
    # str2fun = { 
    #     'art_painting': data_loader.load_pacs,
    #     'cartoon': data_loader.load_pacs,
    #     'photo': data_loader.load_pacs,
    #     'sketch': data_loader.load_pacs,
    #     }   
    columns = ['art_painting', 'cartoon', 'photo', 'sketch']
    target = [i for i in columns if i!=source_domain]
    columns = [source_domain] + target
    print("columns:",columns)
    if eval_mapping:
        index = FA.factor_list
        index.append('w/o do (original x)')
    else:
        index = ['w/o do (original x)']
    index_ours = ['do']
    data_result = {}
    data_result_ours = {}

    for idx, data in enumerate(columns):
        teset = data_loader.load_pacs(data, 'test')
        teloader = DataLoader(teset, batch_size=4, num_workers=0)
        # 计算评价指标
        acc_CA = evaluate_causal(cls_net, teloader, CA, AdaptNet, E_to_W)
        data_result_ours[data] = acc_CA
        #最后一维度是原始数据
        if eval_mapping:
            if data == source_domain:
                teacc_FA_aftermapping, acc_FA = evaluate_mapping(cls_net, teloader, FA, AdaptNet, source=True)
                acc_avg = np.zeros(teacc_FA_aftermapping.shape)
                acc_avg_CA = np.zeros(acc_CA.shape)
            else:
                teacc_FA_aftermapping, acc_FA = evaluate_mapping(cls_net, teloader, FA, AdaptNet, source=False)
                acc_avg = acc_avg + teacc_FA_aftermapping
                acc_avg_CA = acc_avg_CA + acc_CA
            data_result[data]=teacc_FA_aftermapping
            data_result[data+'_FA'] = acc_FA
        else:
            teacc = evaluate(cls_net, teloader)
            if data == source_domain:
                acc_avg = np.zeros(teacc.shape)
                acc_avg_CA = np.zeros(acc_CA.shape)
            else:
                acc_avg = acc_avg + teacc
                acc_avg_CA = acc_avg_CA + acc_CA
            data_result[data] = teacc        
    acc_avg = acc_avg/float(len(target))
    acc_avg_CA = acc_avg_CA/float(len(target))
    
    data_result['Avg'] = acc_avg
    data_result_ours['Avg'] = acc_avg_CA

    df = pd.DataFrame(data_result,index = index)
    df_ours = pd.DataFrame(data_result_ours,index = index_ours)
    print(df)
    print(df_ours)       
    if svpath is not None:
        df.to_csv(svpath)
        df_ours.to_csv(svpath, mode='a')
if __name__=='__main__':
    main()

