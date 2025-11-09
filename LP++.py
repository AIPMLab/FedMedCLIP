import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed
from utils.prepare_data_dg_clip import *
import copy
import argparse
import torch.nn as nn
from nets.models import ClipModelat
import torch.optim as optim
import torch
import numpy as np
from utils.training import train, trainlp, initlp
from utils.testing import test, Glotest, testlp
from utils.aggregation import communication, communication2,communicationlp
from adaptation import LMMDLoss
from adaptation import AdversarialLoss
from nets.models import MLP
from nets.MLPs import ImageMlp, TextMlp
from GetLog import GetLog, GetFileName
import time

def main(Location, dataset, method):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='OfficeHome')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='./data/')
    parser.add_argument('--iters', type=int, default=50,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')
    parser.add_argument('--net', type=str, default='ViT-B/32',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[3])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--step', type=float, default=0)
    parser.add_argument('--aggmode', type=str, default='avg')
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--method', type=str, default='lp')
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lr_mlp', type=float, default=1e-2)
    parser.add_argument('--current_epoch', type=int, default=0)
    parser.add_argument('--shot', type=int, default=16)
    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    args.n_clients = 6
    args = img_param_init(args)
    os.makedirs('./data/', exist_ok=True)
    server_model = ClipModelat(
        args.net, attention=True, freezepy=True)

    train_loaders, val_loaders, test_loaders, test_train, train_test_loaders, Labels = get_data(
        args.dataset)(args, server_model)
    server_model.initdgatal(test_loaders[3])
    client_num = len(test_loaders)
    sclient_num = client_num - len(args.test_envs)
    client_weights = [float(1 / sclient_num) for i in range(sclient_num)]
    client_weights.append(1.)
    models = [copy.deepcopy(server_model).to('cpu') for idx in range(client_num)]
    for i, model in enumerate(models):
        model.setselflabel(Labels[i])
    num_params = sum(p.numel() for p in server_model.parameters())
    mlp = [nn.Linear(512, len(models[idx].labels), bias=True,dtype=models[idx].model.dtype).to(device)  # output is deterimined by how many shots we want. e.g., 5 shots, so it will be num_samples / 5
         for idx in range(client_num)]
    print(f"The network has {num_params} parameters.")
    server_model_pre = server_model
    previous_nets = models
    adv_loss = AdversarialLoss()
    adv = [AdversarialLoss() for idx in range(client_num)]

    net_benefits = GetLog(client_num)
    tpr = GetLog(client_num)
    fpr = GetLog(client_num)
    ValidationLog = GetLog(sclient_num)
    net_benefits_all = []
    net_benefits_none = []
    TestingMetrics = GetLog(client_num)
    Val_Acc = [0 for idx in range(0, sclient_num)]
    Val_mean = 0
    best_testing_acc = [0 for idx in range(0, sclient_num)]  # This only applicable when sharing the MLPs

    if not os.path.exists(Location):
        os.makedirs(Location)

    for i in range(client_num):
        models[i].model.to('cpu')
        models[i].fea_attn.to('cpu')
    alpha_vec = []
    lr_temp = []
    lr_alpha = []
    features = []
    text_weights = []
    labels = []
    for i in range(0, sclient_num):
        models[i].to(device)
        alpha_v, lr_t, lr_al, fea, text_w, la = initlp(args, models[i], train_test_loaders[i], device, mlp[i])
        alpha_vec.append(alpha_v)
        lr_temp.append(lr_t)
        lr_alpha.append(lr_al)
        features.append(fea)
        text_weights.append(text_w)
        labels.append(la)
        models[i].to('cpu')
    # optimizers = [ optim.AdamW(
    #     params=[ {'params': mlp[ idx ].parameters(), 'lr':lr_temp[idx], 'betas': (args.beta1, args.beta2)} ],
    # eps = args.eps, weight_decay = args.weight_decay)
    # for idx in range(sclient_num)]
    optimizers = [ optim.SGD(
        params=[ {'params': mlp[ idx ].parameters(), 'lr': lr_temp[idx]} ], lr=lr_temp[idx], momentum=0.9)
        for idx in range(sclient_num) ]
    schedulers = [ torch.optim.lr_scheduler.ExponentialLR(optimizers[ idx ], gamma=0.97) for idx in
                   range(0, sclient_num) ]

    for a_iter in range(args.iters):  # All Epoch

        time_start = time.time()
        for wi in range(args.wk_iters):  # Each client local training epoch
            print("============ Train epoch {} ============".format(
                wi + a_iter * args.wk_iters))
            for client_idx, model in enumerate(models):
                if client_idx in args.test_envs:
                    pass
                else:
                    models[client_idx].model.to(device)
                    models[client_idx].fea_attn.to(device)
                    mlp[client_idx].to(device)
                    trainlp(
                        args, model, train_test_loaders[client_idx], optimizers[client_idx], device,mlp[client_idx], features[client_idx],
                    text_weights[client_idx], alpha_vec[client_idx],lr_temp[client_idx], labels[client_idx], lr_alpha[client_idx])
                    models[client_idx].model.to('cpu')
                    models[client_idx].fea_attn.to('cpu')
                    mlp[client_idx].to('cpu')
                    schedulers[client_idx].step()
        args.step += 1

        with torch.no_grad():
            server_model.to('cpu')
            mlp[sclient_num].to('cpu')
            mlp = communicationlp(
                args, mlp, client_weights)

            # Validation
            for client_idx in range(client_num):
                if client_idx in args.test_envs:  # Global no validation part
                    pass
                else:
                    models[client_idx].model.to(device)
                    mlp[client_idx].to(device)
                    test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = testlp(args,
                                                                                              models[client_idx],
                                                                                              val_loaders[client_idx],
                                                                                              device,mlp[client_idx],
                                                                                                text_weights[client_idx], alpha_vec[client_idx])
                    models[client_idx].model.to('cpu')
                    models[client_idx].fea_attn.to('cpu')
                    mlp[client_idx].to('cpu')
                    Val_Acc[client_idx] = test_acc
                    if test_acc > best_testing_acc[client_idx]:
                        best_testing_acc[client_idx] = test_acc
                    ValidationLog[client_idx].append(
                        [100. * test_acc, 100. * bacc, 100. * f1, 100. * precision, 100. * recall])
                    print(' Test site-{:d}| Validation Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))

            time_end = time.time()
            print(time_end-time_start)

            if (sum(Val_Acc)) / sclient_num > Val_mean:
                Val_mean = (sum(Val_Acc)) / sclient_num
                for i in range(0, sclient_num):
                    torch.save(mlp[i].state_dict(),
                               Location + 'lp' + str(i) + '.pth')

    # Testing
    with torch.no_grad():
        for i in range(0, sclient_num):
            mlp[i].load_state_dict(torch.load(Location + 'lp' + str(i) + '.pth'))
        mlp[sclient_num].to('cpu')
        mlp = communicationlp(
            args, mlp, client_weights)

    for client_idx in range(client_num):

        if client_idx in args.test_envs:
            # pass
            server_model.to('cuda')
            mlp[client_idx].to('cuda')
            test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = testlp(args, server_model,
                                                                                         test_loaders[client_idx],
                                                                                         device,mlp[client_idx],
                                                                                         text_weights[0], alpha_vec[0])
            server_model.to('cpu')
            mlp[client_idx].to('cpu')
            net_benefits[client_idx] = benefits
            tpr[client_idx] = tp
            fpr[client_idx] = fp
            net_benefits_all = all
            net_benefits_none = none
            TestingMetrics[sclient_num].append(
                [100. * test_acc, 100. * bacc, 100. * f1, 100. * precision, 100. * recall])
            print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))
        else:
            models[client_idx].model.to(device)
            mlp[client_idx].to(device)
            test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = testlp(args, models[client_idx],
                                                                                      test_loaders[client_idx],
                                                                                      device,mlp[client_idx],
                                                                                         text_weights[client_idx], alpha_vec[client_idx])
            models[client_idx].model.to('cpu')
            models[client_idx].fea_attn.to('cpu')
            mlp[client_idx].to('cpu')
            net_benefits[client_idx] = benefits
            tpr[client_idx] = tp
            fpr[client_idx] = fp
            TestingMetrics[client_idx].append([100. * test_acc, 100. * bacc, 100. * f1, 100. * precision, 100. * recall])
            print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f}'.format(client_idx, test_acc, bacc))

    ValidationFile = GetFileName('ValidationMetrics', sclient_num)  # ValidationMetrics(i).csv
    NetBenefitFile = GetFileName('NetBenefits', client_num)
    NetBenefitALLFile = GetFileName('NetBenefitsALL', 1)
    tprFile = GetFileName('TPR', client_num)
    fprFile = GetFileName('FPR', client_num)
    TestingMetricsFile = GetFileName('TestingMetrics', client_num)
    logs = [ValidationLog, net_benefits, tpr, fpr, net_benefits_all, TestingMetrics]
    FileName = ['ValidationFile', 'NetBenefitFile', 'tprFile', 'fprFile', 'net_all_none', 'TestingMetricsFile']
    if not os.path.exists('./results' + dataset + method ):
        os.makedirs('./results' + dataset + method )
    for j in FileName:
        if j == 'ValidationFile':
            print(ValidationLog)
            for m in range(0, sclient_num):
                np_log = np.array(ValidationLog[m], dtype=float)  # ValidationLog is a list []
                loc = './results' + dataset + method  + ValidationFile[m]  # location
                np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')
        if j == 'NetBenefitFile':
            for m in range(0, client_num):
                np_log = np.array(net_benefits[m], dtype=float)
                loc = './results' + dataset + method   + NetBenefitFile[m]
                np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')
        if j == 'tprFile':
            for m in range(0, client_num):
                np_log = np.array(tpr[m], dtype=float)
                loc = './results' + dataset + method  + tprFile[m]
                np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')

        if j == 'fprFile':
            for m in range(0, client_num):
                np_log = np.array(fpr[m], dtype=float)
                loc = './results' + dataset + method  + fprFile[m]
                np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')

        if j == 'net_all_none':
            for m in range(0, 1):
                np_log = np.array(net_benefits_all, dtype=float)
                loc = './results' + dataset + method  + NetBenefitALLFile[m]
                np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')

        if j == 'TestingMetricsFile':
            for m in range(0, client_num):
                np_log = np.array(TestingMetrics[m], dtype=float)
                loc = './results' + dataset + method  + TestingMetricsFile[m]
                np.savetxt(loc, np_log, delimiter=',', fmt='%.6f')


if __name__ == '__main__':
    Location = './SavedModel/OfficeHome/ACPR/LP++/RN50/lr=5e-4/'
    method = '/LP++/RN50/lr=5e-5/'
    dataset = '/OfficeHome/ACPR/'
    main(Location, dataset, method)
