import torch
from opacus import  PrivacyEngine
import math
import copy
from utils.training import weight_flatten, set_parameters
from utils.clip_util import compress_model, decompress_model
from nets.models import ClipModelat_quantize


def e(x):
    sigma = 1.0
    return math.exp(-x / sigma) / sigma

def communication(args, server_model, models, client_weights):
    client_num = len(models)
    # server_model = Privacy.make_private(module=server_model, noise_multiplier=1.1, max_grad_norm=1.0)
    if args.aggmode == 'att':
        with torch.no_grad():
            for key in server_model.fea_attn.state_dict().keys():
                # print(key)
                # if key == '1.weight' or key == '1.bias' or key == '1.running_mean'\
                #         or key == '1.running_var':
                #     pass
                if 'num_batches_tracked' in key or 'bert' in key:
                    server_model.fea_attn.state_dict()[key].data.copy_(
                    models[client_num-1].fea_attn.state_dict()[key]) # model[3] means the index of global model, depending on your settings.
                else:
                    temp = torch.zeros_like(server_model.fea_attn.state_dict()[
                                                key ], dtype=torch.float32)
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            temp += client_weights[ client_idx ] * \
                                    models[ client_idx ].fea_attn.state_dict()[ key ]
                    server_model.fea_attn.state_dict()[ key ].data.copy_(temp)
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            models[ client_idx ].fea_attn.state_dict()[ key ].data.copy_(
                                server_model.fea_attn.state_dict()[ key ])
            # for key in mlp[3].state_dict().keys():
            #     if 'num_batches_tracked' in key or 'bert' in key:
            #         mlp[3].state_dict()[key].data.copy_(
            #         mlp[3].state_dict()[key]) # model[3] means the index of global model, depending on your settings.
            #     else:
            #         temp = torch.zeros_like(mlp[3].state_dict()[
            #                                     key ], dtype=torch.float32)
            #         for client_idx in range(client_num):
            #             if client_idx not in args.test_envs:
            #                 temp += client_weights[ client_idx ] * \
            #                         mlp[ client_idx ].state_dict()[ key ]
            #         mlp[3].state_dict()[ key ].data.copy_(temp)
            #         for client_idx in range(client_num):
            #             if client_idx not in args.test_envs:
            #                 mlp[ client_idx ].state_dict()[ key ].data.copy_(
            #                     mlp[3].state_dict()[ key ])

    if args.aggmode == 'avg':
        with torch.no_grad():
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key or 'bert' in key:
                    server_model.state_dict()[key].data.copy_(
                        models[client_num-1].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[
                                                key], dtype=torch.float32)
                    for client_idx in range(client_num):  # aggregation
                        if client_idx not in args.test_envs:
                            temp += client_weights[client_idx] * \
                                    models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):  # broadcast
                        if client_idx not in args.test_envs:
                            models[client_idx].state_dict()[key].data.copy_(
                                server_model.state_dict()[key])
    return server_model, models



def communication2(args, client_weights, mlp):
    client_num = len(mlp)
    with torch.no_grad():
        for key in mlp[client_num-1].layers.state_dict().keys():
            temp = torch.zeros_like(mlp[client_num-1].layers.state_dict()[
                                        key], dtype=torch.float32)
            for client_idx in range(client_num):
                if client_idx not in args.test_envs:
                    temp += client_weights[client_idx] * \
                            mlp[client_idx].layers.state_dict()[key]
            mlp[client_num-1].layers.state_dict()[key].data.copy_(temp)
            for client_idx in range(client_num):
                if client_idx not in args.test_envs:
                    mlp[client_idx].layers.state_dict()[key].data.copy_(
                        mlp[client_num-1].layers.state_dict()[key])

    return mlp


def comunicationfedamp(args, server_model, models):
    client_num = len(models)
    alphaK = 1.0
    with torch.no_grad():
        for c in range(0, client_num):
            mu = copy.deepcopy(server_model) # server model copy
            for param in mu.parameters():
                param.data.zero_() # zero_grad
            coef = torch.zeros(client_num) # coefficient matrix
            for j, mw in enumerate(models):
                if c != j:
                    weights_i = weight_flatten(models[c].model) # Client c weights
                    weights_j = weight_flatten(mw.model) # Client j weights, c != j
                    sub = (weights_i - weights_j).view(-1) # difference
                    sub = torch.dot(sub, sub)
                    coef[j] = alphaK * e(sub)
                else:
                    coef[j] = 0
            print(coef)
            coef_self = 1 - torch.sum(coef)
            print(coef_self)
            for j, mw in enumerate(models):
                for param, param_j in zip(mu.parameters(), mw.parameters()): # change params for server_model with Clients
                    param.data += coef[j] * param_j # temp params learned from each client

            set_parameters(models[c], mu, coef_self) # Client i current model, previous model, coef_self

    return models


def communicationfedrep(args, server_model, models, client_weights):
    client_num = len(models)
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key or 'bert' in key:
                server_model.state_dict()[key].data.copy_(
                    models[client_num - 1].state_dict()[key])
            elif 'model.visual.ln_post.weight' in key or 'model.visual.ln_post.bias' in key or 'model.ln_final.weight'\
                    in key or 'model.ln_final.bias' in key:
                continue # sharing only feature representation, keep last layer locally
            else:
                temp = torch.zeros_like(server_model.state_dict()[
                                            key], dtype=torch.float32)
                for client_idx in range(client_num):  # aggregation
                    if client_idx not in args.test_envs:
                        temp += client_weights[client_idx] * \
                                models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):  # broadcast
                    if client_idx not in args.test_envs:
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key])
    return server_model, models


# def communicationpromptFL(args, mlp, client_weights):
#     client_num = len(mlp)
#     with torch.no_grad():
#         for key in mlp[ client_num - 1 ].state_dict().keys():
#             temp = torch.zeros_like(mlp[ client_num - 1 ].state_dict()[
#                                         key ], dtype=torch.float32)
#             for client_idx in range(client_num):
#                 if client_idx not in args.test_envs:
#                     temp += client_weights[ client_idx ] * \
#                             mlp[ client_idx ].state_dict()[ key ]
#             mlp[ client_num - 1 ].state_dict()[ key ].data.copy_(temp)
#             for client_idx in range(client_num):
#                 if client_idx not in args.test_envs:
#                     mlp[ client_idx ].state_dict()[ key ].data.copy_(
#                         mlp[ client_num - 1 ].state_dict()[ key ])
#
#     return mlp

def communicationpromptFL(args, mlp, client_weights):
    client_num = len(mlp) # 6
    with torch.no_grad():
        for key in mlp[ client_num - 1 ].state_dict().keys():
            if 'ctx' in key or 'token_prefix' in key or 'token_suffix' in key:
            # if '!' in key or '!' in key:
                pass
            else:
                print(key)
                temp = torch.zeros_like(mlp[client_num - 1].state_dict()[
                                            key], dtype=torch.float32)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        temp += client_weights[client_idx] * \
                                mlp[client_idx].state_dict()[key]
                mlp[client_num - 1].state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        mlp[client_idx].state_dict()[key].data.copy_(
                            mlp[client_num - 1].state_dict()[key])

    return mlp

def communicationCocoOp(args, mlp, client_weights):
    client_num = len(mlp) # clients + global, for isic it is 7, but index is [0, 6]
    with torch.no_grad():
        for key in mlp[ client_num - 1 ].state_dict().keys():
            if 'ctx' in key or 'token_prefix' in key or 'token_suffix' in key:
            # if '!' in key or '!' in key:
                pass
            else:
                temp = torch.zeros_like(mlp[client_num - 1].state_dict()[
                                            key], dtype=torch.float32)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        temp += client_weights[client_idx] * \
                                mlp[client_idx].state_dict()[key]
                mlp[client_num - 1].state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        mlp[client_idx].state_dict()[key].data.copy_(
                            mlp[client_num - 1].state_dict()[key])

    return mlp

def communicationlp(args, mlp, client_weights):
    client_num = len(mlp)
    with torch.no_grad():
        for key in mlp[ client_num - 1 ].state_dict().keys():
            print(key)
            # if 'weight' in key or 'bias' in key :
            if '!' in key or '!' in key:
                pass
            else:
                temp = torch.zeros_like(mlp[ client_num - 1 ].state_dict()[
                                            key ], dtype=torch.float32)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        temp += client_weights[ client_idx ] * \
                                mlp[ client_idx ].state_dict()[ key ]
                mlp[ client_num - 1 ].state_dict()[ key ].data.copy_(temp)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        mlp[ client_idx ].state_dict()[ key ].data.copy_(
                            mlp[ client_num - 1 ].state_dict()[ key ])

    return mlp

def communicationmlp(args, server_model, models, client_weights):
    client_num = len(models)
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key or 'bert' in key:
                server_model.state_dict()[key].data.copy_(
                    models[client_num - 1].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[
                                            key], dtype=torch.float32)
                for client_idx in range(client_num):  # aggregation
                    if client_idx not in args.test_envs:
                        temp += client_weights[client_idx] * \
                                models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):  # broadcast
                    if client_idx not in args.test_envs:
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key])
    return server_model, models


def paramsReusing(args, server_model, mlps):
    for mlp in mlps:
        k = 0
        with torch.no_grad():
            for key, key2 in zip(mlp.state_dict().keys(), server_model.fea_attn.state_dict().keys()):
                # if 'running_mean' not in key or 'running_var' not in key or 'num_batches_tracked' not in key:
                #     mlp.state_dict()[key].data.copy_(server_model.fea_attn.state_dict()[key2])
                if 'layers.0' in key:
                    mlp.state_dict()[key].data.copy_(server_model.fea_attn.state_dict()[key2])
    return mlps


def communication_quantized(args, server_model, models, client_weights):
    client_num = len(models)
    ### dequantize
    server_model = decompress_model(server_model, ClipModelat_quantize(args.net, freezepy=True))
    # print(server_model, server_model_clone)

    for client_idx in range(client_num):
        models[client_idx] = decompress_model(models[client_idx], ClipModelat_quantize(args.net, freezepy=True))

    if args.aggmode == 'att':
        with torch.no_grad():
            for key in server_model.fea_attn.state_dict().keys():
                if 'num_batches_tracked' in key or 'bert' in key:
                    server_model.fea_attn.state_dict()[key].data.copy_(
                    models[client_num-1].fea_attn.state_dict()[key]) # model[3] means the index of global model, depending on your settings.
                else:
                    temp = torch.zeros_like(server_model.fea_attn.state_dict()[
                                                key ], dtype=torch.float32)
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            temp += client_weights[ client_idx ] * \
                                    models[ client_idx ].fea_attn.state_dict()[ key ]
                    server_model.fea_attn.state_dict()[ key ].data.copy_(temp)
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            models[ client_idx ].fea_attn.state_dict()[ key ].data.copy_(
                                server_model.fea_attn.state_dict()[ key ])

        server_model = compress_model(server_model)
        for client_idx in range(client_num):
            models[client_idx] = compress_model(models[client_idx])

    return server_model, models
