import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.sot_model as sot_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU



SOT_CONFIG = {
    'search_feature_sz': 18,
    'template_feature_sz': 8,
    'search_sz': 18 * 16,
    'temp_sz': 8 * 16}

def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'SOT with default settings.'


    settings.batch_size = 12
    settings.num_workers = 2
    settings.multi_gpu = False

    # settings.batch_size = 38
    # settings.num_workers = 4
    # settings.multi_gpu = True

    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
#   输入输出 尺寸参数
    # settings.search_feature_sz的作用域：仅用于数据预处理阶段确定输入图像尺寸 不会传递到特征融合网络仅影响：search_sz = 32×8 = 256，决定输入图像大小
    settings.search_feature_sz = 18
    settings.template_feature_sz = 8
    settings.search_sz = settings.search_feature_sz * 16
    settings.temp_sz = settings.template_feature_sz * 16

    # settings.search_feature_sz = 32
    # settings.template_feature_sz = 16
    # settings.search_sz = settings.search_feature_sz * 8
    # settings.temp_sz = settings.template_feature_sz * 8

#抖动因子:人为制造定位误差和尺度误差
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}
#方案2
    # settings.center_jitter_factor = {'search': 3, 'template': 0.5}
    # settings.scale_jitter_factor = {'search': 0.25, 'template': 0.15}
#方案三
    # settings.center_jitter_factor = {'search': 3, 'template': 0.7}
    # settings.scale_jitter_factor = {'search': 0.25, 'template': 0.2}

    # Transformer
    settings.position_embedding = 'sine'        #位置编码
    settings.hidden_dim = 256
    settings.dropout = 0.1                      #增强鲁棒性
    # settings.dropout = 0.08
    settings.nheads = 8

    # settings.dim_feedforward = 2048
    settings.dim_feedforward = 1024
#dim_feedforward是前馈子网络（FFN）中间层的扩维倍数，通常是四倍

    settings.featurefusion_layers = 2            #头数


    # Train datasets
    # lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir,version="2017")

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.SOTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)


    #    总共训练 1000 个 epoch，每个 epoch 包含 1000 次迭代。
    #    学习率在 第 500 个 epoch 后下降为原来的 1/10
    # The sampler for training
    #只用got10k训练
    dataset_train = sampler.SOTSampler([got10k_train], [1],
                                samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)
#用三数据集训练
    # dataset_train = sampler.SOTSampler([lasot_train, got10k_train, coco_train], [1,1,1],
    #                             samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)
#coco+got10k
    # dataset_train = sampler.SOTSampler([got10k_train, coco_train], [1,1],
    #                             samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)

    # dataset_train = sampler.SOTSampler([lasot_train, got10k_train, coco_train, trackingnet_train], [1,1,1,1],
    #                             samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)
    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)

    # Create network and actor
    model = sot_models.sot_model(settings)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = sot_models.sot_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.SOTActor(net=model, objective=objective)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,
                                  weight_decay=1e-4)

# 训练到第 500 个 epoch 时，把学习率乘以 0.1（降到原来的 10%），之后继续训练。
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 300)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 400)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    #训练
    trainer.train(500, load_latest=True, fail_safe=True)
