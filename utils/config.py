

import random
import numpy as np
import torch


def img_param_init(args):
    dataset = args.dataset
    # if dataset =='Prostate':
    #     domains = ['Center_1', 'Center_2', 'Center_3', 'Center_4']
    # if dataset =='Prostate':
    #     domains = ['Center_4', 'Center_1', 'Center_2', 'Center_3']
    # if dataset =='Prostate':
    #     domains = ['Center_3','Center_4', 'Center_1', 'Center_2']
    if dataset =='Prostate':
        domains = ['Center_2','Center_3','Center_4', 'Center_1']
    if dataset =='brain_ICH':
        domains = ['./divided(10clients)/client_' + str(idx) for idx in range(0, 10)]
        domains.append('./divided(20clients)/Global')
    if dataset =='BrainTumor':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BraTS':
        domains = ['client_0', 'client_1', 'client_2', 'client_3','Global']
    if dataset =='BT4':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BT2':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BT_iid':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='ISIC2019':
        domains = ['anterior torso', 'head or neck', 'lower extremity', 'palms or soles','posterior torso','upper extremity','Nan']
    if dataset =='ISIC2019_2':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='BT_Large':
        domains = ['client_0', 'client_1', 'client_2', 'client_3','client_4','client_5']
    if dataset =='Alzheimer':
        domains = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5','client_6', 'client_7', 'client_8', 'client_9']
    # if dataset =='ModernOffice31':
    #     domains = ['a','d','s','w']
    if dataset =='ModernOffice31':
        domains = ['d','s','w','a']
    # if dataset =='ModernOffice31':
    #     domains = ['s', 'w', 'a', 'd']
    # if dataset =='ModernOffice31':
    #     domains = ['w','a','d','s']
    if dataset =='MultiImageNet':
        domains = ['M','R','S','T']
    if dataset =='Multi_source':
        domains = ['a','d','w','OA','OC','OP','OR','s']
    if dataset =='RealSkin':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    if dataset =='DomainNet':
        domains = ['C', 'I', 'P', 'Q', 'R','S' ]
    # if dataset =='DomainNet':
    #     domains = ['S','C', 'I', 'P', 'Q', 'R' ]
    # if dataset =='DomainNet':
    #     domains = ['R','S','C', 'I', 'P', 'Q' ]
    # if dataset =='DomainNet':
    #     domains = ['Q','R','S','C', 'I', 'P' ]
    # if dataset =='DomainNet':
    #     domains = ['P','Q','R','S','C', 'I' ]
    # if dataset =='DomainNet':
    #     domains = ['I','P','Q','R','S','C' ]
    # if dataset =='DomainNet':
    #     domains = ['C', 'I', 'P', 'Q', 'R', 'S']

    # if dataset =='DomainNet':
    #     domains = ['Q','P','I', 'C', 'S', 'R' ]
    # if dataset =='DomainNet':
    #     domains0 = ['./Scalability/C/client_' + str(idx) for idx in range(0, 10)]
    #     domains2 =['./Scalability/I/client_' + str(idx) for idx in range(0, 10)]
    #     domains3 = ['./Scalability/P/client_' + str(idx) for idx in range(0, 10)]
    #     domains4 = ['./Scalability/Q/client_' + str(idx) for idx in range(0, 10)]
    #     domains5 = ['./Scalability/R/client_' + str(idx) for idx in range(0, 10)]
    #     domains = domains0 + domains2 + domains3 + domains4 + domains5
    #     domains.append('S')
    if dataset == 'miniDomainNet':
        domains = ['C', 'I', 'P', 'Q', 'R', 'S']
    # if dataset =='miniDomainNet':
    #     domains = ['S','C', 'I', 'P', 'Q', 'R' ]
    # if dataset =='miniDomainNet':
    #     domains = ['R','S','C', 'I', 'P', 'Q' ]
    # if dataset =='miniDomainNet':
    #     domains = ['Q','R','S','C', 'I', 'P' ]
    # if dataset =='miniDomainNet':
    #     domains = ['P','Q','R','S','C', 'I' ]
    # if dataset =='miniDomainNet':
    #     domains = ['I','P','Q','R','S','C' ]

    if dataset =='havior':
        domains = ['client_0', 'client_1', 'client_2', 'client_3']
    # if dataset =='OfficeHome':
    #     domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    # if dataset == 'OfficeHome':
    #     domains = ['./MultiClients/client_' + str(idx) for idx in range(0, 15)]
    #     domains.append('R')
    if dataset =='OfficeHome':
        domains = ['A', 'C', 'P', 'R' ]
    # if dataset =='OfficeHome':
    #     domains = ['C', 'P', 'R', 'A' ]
    # if dataset =='OfficeHome':
    #     domains = [ 'A','P', 'R', 'C' ]
    # if dataset =='OfficeHome':
    #     domains = ['A', 'C', 'R', 'P' ]
    if dataset =='SkinCen':
        domains = ['client_0', 'client_1']
    if dataset =='LargeImage':
        domains = ['defocus_blur1', 'defocus_blur2', 'defocus_blur3', 'defocus_blur4', 'defocus_blur5', 'glass_blur1', 'glass_blur2', 'glass_blur3', 'glass_blur4', 'glass_blur5', 'motion_blur1', 'motion_blur2', 'motion_blur3', 'motion_blur4', 'motion_blur5', 'zoom_blur1', 'zoom_blur2', 'zoom_blur3', 'zoom_blur4', 'zoom_blur5', 'contrast1', 'contrast2', 'contrast3', 'contrast4', 'contrast5', 'elastic_transform1', 'elastic_transform2', 'elastic_transform3', 'elastic_transform4', 'elastic_transform5', 'jpeg_compression1', 'jpeg_compression2', 'jpeg_compression3', 'jpeg_compression4', 'jpeg_compression5', 'pixelate1', 'pixelate2', 'pixelate3', 'pixelate4', 'pixelate5', 'gaussian_blur1', 'gaussian_blur2', 'gaussian_blur3', 'gaussian_blur4', 'gaussian_blur5', 'saturate1', 'saturate2', 'saturate3', 'saturate4', 'saturate5', 'spatter1', 'spatter2', 'spatter3', 'spatter4', 'spatter5', 'speckle_noise1', 'speckle_noise2', 'speckle_noise3', 'speckle_noise4', 'speckle_noise5', 'gaussian_noise1', 'gaussian_noise2', 'gaussian_noise3', 'gaussian_noise4', 'gaussian_noise5', 'impulse_noise1', 'impulse_noise2', 'impulse_noise3', 'impulse_noise4', 'impulse_noise5', 'shot_noise1', 'shot_noise2', 'shot_noise3', 'shot_noise4', 'shot_noise5', 'brightness1', 'brightness2', 'brightness3', 'brightness4', 'brightness5', 'fog1', 'fog2', 'fog3', 'fog4', 'fog5', 'frost1', 'frost2', 'frost3', 'frost4', 'frost5', 'snow1', 'snow2', 'snow3', 'snow4', 'snow5', 'global']
    # if dataset =='LargeImage':
    #     domains = ['defocus_blur1', 'defocus_blur2', 'defocus_blur3', 'defocus_blur4', 'defocus_blur5', 'glass_blur1']
    if dataset == 'Food101':
        domains = ['./pathological/client_' + str(idx) for idx in range(0, 20)]
        domains.append('./pathological/Global')

    if dataset =='DTD':
        domains = ['./practical/client_'+str(idx) for idx in range(0, 10)]
        domains.append('./practical/Global')

    args.domains = domains
    if args.dataset =='BrainTumor':
        args.num_classes = 4
    if args.dataset =='ISIC2019':
        args.num_classes = 6
    if args.dataset =='ISIC2019_2':
        args.num_classes = 7
    if args.dataset =='Prostate':
        args.num_classes = 2
    if args.dataset =='miniDomainNet':
        args.num_classes = 65
    if args.dataset =='DTD':
        args.num_classes = 47
    if args.dataset =='brain_ICH':
        args.num_classes = 5
    if args.dataset =='Food101':
        args.num_classes = 101
    if args.dataset =='Multi_source':
        args.num_classes = 4
    if args.dataset =='BT4':
        args.num_classes = 4
    if args.dataset =='BT2':
        args.num_classes = 4
    if args.dataset =='BT_iid':
        args.num_classes = 4
    if args.dataset =='BT_Large':
        args.num_classes = 4
    if args.dataset =='havior':
        args.num_classes = 23
    if args.dataset =='Alzheimer':
        args.num_classes = 4
    if args.dataset =='MultiImageNet':
        args.num_classes = 200
    if args.dataset =='Skin':
        args.num_classes = 9
    if args.dataset =='Skin2':
        args.num_classes = 9
    if args.dataset =='BraTS':
        args.num_classes = 2
    if args.dataset =='Skin_Large':
        args.num_classes = 6
    if args.dataset =='ModernOffice31':
        args.num_classes = 31
    if args.dataset =='Skin4':
        args.num_classes = 9
    if args.dataset =='BT44':
        args.num_classes = 14
    if args.dataset =='RealSkin':
        args.num_classes = 7
    if args.dataset =='SkinCen':
        args.num_classes = 7
    if args.dataset =='OfficeHome':
        args.num_classes = 65
    if args.dataset =='DomainNet':
        args.num_classes = 345
    if args.dataset =='LargeImage':
        args.num_classes = 1000
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
