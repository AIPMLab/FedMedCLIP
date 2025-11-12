import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed
from utils.prepare_data_dg_clip import *
import copy
import argparse
from nets.models import ClipModelat, ClipModelat_quantize
import torch.optim as optim
import torch
import numpy as np
from utils.training import train
from utils.testing import test, Glotest
from utils.aggregation import communication, communication2, paramsReusing, communication_quantized
from adaptation import LMMDLoss
from adaptation import AdversarialLoss
from nets.models import MLP, MLP_old
from nets.MLPs import ImageMlp, TextMlp
from utils.clip_util import freeze_param_mlp, compress_model, decompress_model
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
    parser.add_argument('--aggmode', type=str, default='att')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--method', type=str, default='attn_mlp')
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_mlp', type=float, default=1e-4)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1)
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
    print(f"The network has {num_params} parameters.")
    server_model_pre = server_model
    previous_nets = models
    adv_loss = AdversarialLoss()
    adv = [AdversarialLoss() for idx in range(client_num)]
    mlp = [MLP(hidden_size=512, input_size=512, num_classes=len(Labels[idx])) for idx in range(client_num)]

    num_params = sum(p.numel() for p in mlp[0].parameters())
    # print(mlp[0])
    num_params_trainbale = sum(p.numel() for p in mlp[0].parameters() if p.requires_grad)
    print(f"The MLP has {num_params} parameters and {num_params_trainbale} trainable params.")
    for name, param in mlp[0].named_parameters():
        if param.requires_grad:
            print(f"{name} requires grad")
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

    # schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizers[idx], gamma=0.97) for idx in range(0, sclient_num)]
    torch.save(mlp[sclient_num].state_dict(), Location + 'mlp' + str(sclient_num) + '.pth')
    optimizers_tta = [optim.SGD(
        params=[
                {'params': mlp[idx].parameters(), 'lr': 1e-3}],momentum=0.9, lr=1e-3)
        for idx in range(client_num)]

    for i in range(client_num):
        models[i].model.to('cpu')
        models[i].fea_attn.to('cpu')
        mlp[i].to('cpu')
        server_model.to('cpu')

    for a_iter in range(args.iters):  # All Epoch
        optimizers = [optim.AdamW(
            params=[{'params': models[idx].fea_attn.parameters(), 'lr': args.lr, 'betas': (args.beta1, args.beta2)},
                    {'params': mlp[idx].parameters(), 'lr': args.lr_mlp, 'betas': (args.beta1, args.beta2)}],
            eps=args.eps, weight_decay=args.weight_decay)
            for idx in range(client_num)]
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
                    train(
                        args, model, train_test_loaders[client_idx], optimizers[client_idx], device,
                        test_train[0], adv[client_idx], server_model_pre, previous_nets[client_idx],
                        mlp[client_idx])
                    models[client_idx].model.to('cpu')
                    models[client_idx].fea_attn.to('cpu')
                    mlp[client_idx].to('cpu')
    
        args.lr = args.lr * 0.97
        args.lr_mlp = args.lr_mlp * 0.97
        args.step += 1
    
        with torch.no_grad():
            server_model_copy = copy.deepcopy(server_model)
            server_model = compress_model(server_model)
            for client_idx in range(client_num):
                models[client_idx] = compress_model(models[client_idx])
    
            server_model, models = communication_quantized(
                args, server_model, models, client_weights)
            # server_model = decompress_model(server_model, ClipModelat_quantize(args.net, freezepy=True))
            server_model = decompress_model(server_model, server_model_copy)
            # server_model.to('cpu')
            for client_idx in range(client_num):
                models[client_idx] = decompress_model(models[client_idx], server_model_copy)
                # models[client_idx].to('cpu')
            for i, model in enumerate(models):
                model.setselflabel(Labels[i])
            server_model.setselflabel(Labels[client_num-1])
    
            # Validation
            for client_idx in range(client_num):
                if client_idx in args.test_envs:  # Global no validation part
                    pass
                else:
                    models[client_idx].model.to(device)
                    models[client_idx].fea_attn.to(device)
                    mlp[client_idx].to(device)
                    test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = test(args,
                                                                                              models[client_idx],
                                                                                              val_loaders[client_idx],
                                                                                              device, mlp[client_idx])
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
            print(time_end - time_start)
    
            if (sum(Val_Acc)) / sclient_num > Val_mean:
                Val_mean = (sum(Val_Acc)) / sclient_num
                for i in range(0, sclient_num):
                    torch.save(models[i].fea_attn.state_dict(),
                               Location + 'Client' + str(i) + '.pth')
                    torch.save(compress_model(copy.deepcopy(models[i])),
                               Location + 'Client_quantized_' + str(i) + '.pth')
                    torch.save(mlp[i].state_dict(), Location + 'mlp' + str(i) + '.pth')

    # Testing
    with torch.no_grad():
        for i in range(0, sclient_num):
            models[i].fea_attn.load_state_dict(torch.load(Location + 'Client' + str(i) + '.pth'))
            mlp[i].load_state_dict(torch.load(Location + 'mlp' + str(i) + '.pth'))
        server_model, models = communication(
            args, server_model, models, client_weights)
        for i in range(0, sclient_num):
            torch.save(models[i].fea_attn.state_dict(),
                       Location + 'Client' + str(i) + '.pth')
        # mlp = communication2(args, client_weights, mlp)
        # for i in range(0, client_num):
        #     torch.save(mlp[i].state_dict(), Location + 'mlp' + str(i) + '.pth')

    for client_idx in range(client_num):
        args.index = client_idx
        if client_idx in args.test_envs:
            server_model.to('cuda')
            # mlp[client_idx].load_state_dict(torch.load(Location + 'mlp' + str(client_idx) + '.pth'))
            # mlp[client_idx].to('cuda')
            test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = Glotest(args, server_model,
                                                                                         test_loaders[client_idx],
                                                                                         device)
            server_model.to('cpu')
            net_benefits[client_idx] = benefits
            tpr[client_idx] = tp
            fpr[client_idx] = fp
            net_benefits_all = all
            net_benefits_none = none
            TestingMetrics[sclient_num].append(
                [100. * test_acc, 100. * bacc, 100. * f1, 100. * precision, 100. * recall])
            print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f} | F1: {:.4f}'.format(client_idx, test_acc, bacc, f1))
        else:
            mlp[client_idx].load_state_dict(torch.load(Location + 'mlp' + str(client_idx) + '.pth'))
            models[client_idx].model.to(device)
            models[client_idx].fea_attn.to(device)
            mlp[client_idx].to(device)
            test_acc, bacc, f1, precision, recall, benefits, tp, fp, all, none = test(args, models[client_idx],
                                                                                      test_loaders[client_idx],
                                                                                      device,
                                                                                      mlp[client_idx], optimizer_tta=optimizers_tta[client_idx])
            models[client_idx].model.to('cpu')
            models[client_idx].fea_attn.to('cpu')
            mlp[client_idx].to('cpu')
            net_benefits[client_idx] = benefits
            tpr[client_idx] = tp
            fpr[client_idx] = fp
            TestingMetrics[client_idx].append([test_acc, bacc, f1, precision, recall])
            print(' Test site-{:d}| Test Acc: {:.4f} | Bacc: {:.4f} | F1: {:.4f}'.format(client_idx, test_acc, bacc, f1))

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
    # This only used for loss_sim 0.01
    Location = './SavedModel/OfficeHome/AAAI2026/ACPR/Fed_attnMLP/ViT-B-32/lr=5e-5/twolayerMLP(lr=1e-3)/JSBased/Quantized/'
    method = '/Fed_attnMLP/ViT-B-32/lr=5e-5/twolayerMLP(lr=1e-3)/JSBased/Quantized/'
    dataset = '/OfficeHome/AAAI2026/ACPR/'
    main(Location, dataset, method)
