import argparse
import datetime
import json

import numpy as np
import os
import time
from pathlib import Path

import paddle
import paddle.distributed as dist
from visualdl import LogWriter

import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from huggingface_hub import hf_hub_download, login
from engine_finetune import train_one_epoch, evaluate, Mixup

import warnings
import faulthandler

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='/home/chenlb24/RETFound_MAE/RETFound_mae_meh_paddle.pdparams', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/chenlb24/RETFound_MAE/data/APTOS2019', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=5, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='gpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # fine-tuning parameters
    parser.add_argument('--savemodel', action='store_true', default=True,
                        help='Save model')
    parser.add_argument('--norm', default='IMAGENET', type=str, help='Normalization method')
    parser.add_argument('--enhance', action='store_true', default=False, help='Use enhanced data')
    parser.add_argument('--datasets_seed', default=2026, type=int)

    return parser


def main(args, criterion):
    if args.resume and not args.eval:
        resume = args.resume
        checkpoint = paddle.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        args = checkpoint['args']
        args.resume = resume

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = paddle.device.set_device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    paddle.seed(seed)
    np.random.seed(seed)

    if args.model=='RETFound_mae':
        model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )
    
    if args.finetune and not args.eval:
        
        print(f"Loading pre-trained PaddlePaddle weights from: {args.finetune}")
        
        # 直接加载PaddlePaddle格式的权重
        if not os.path.exists(args.finetune):
            raise FileNotFoundError(f"PaddlePaddle weights file not found: {args.finetune}")
        
        checkpoint_model = paddle.load(args.finetune)
        print("Load pre-trained PaddlePaddle checkpoint from: %s" % args.finetune)
        
        # 检查权重文件格式，支持直接的state_dict或包含'model'键的格式
        if isinstance(checkpoint_model, dict) and 'model' in checkpoint_model:
            checkpoint_model = checkpoint_model['model']
            
        # 通用的权重加载逻辑
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding (对PaddlePaddle权重也适用)
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.set_state_dict(checkpoint_model)
        print(f"Loading result: {msg}")

        # 重新初始化分类头
        paddle.nn.initializer.Normal(model.head.weight, std=2e-5)

    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)

    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = LogWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    if not args.eval:
        data_loader_train = paddle.io.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        print(f'len of train_set: {len(data_loader_train) * args.batch_size}')

        data_loader_val = paddle.io.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )

    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.resume and args.eval:
        checkpoint = paddle.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        model.set_state_dict(checkpoint['model'])

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=no_weight_decay,
                                        layer_decay=args.layer_decay
                                        )
    optimizer = paddle.optimizer.AdamW(parameters=param_groups, learning_rate=args.lr)
    loss_scaler = NativeScaler()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        if 'epoch' in checkpoint:
            print("Test with the best model at epoch = %d" % checkpoint['epoch'])
        test_stats, auc_roc = evaluate(data_loader_test, model, device, args, epoch=0, mode='test',
                                       num_class=args.nb_classes, log_writer=log_writer)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        val_stats, val_score = evaluate(data_loader_val, model, device, args, epoch, mode='val',
                                        num_class=args.nb_classes, log_writer=log_writer)
        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))


        if epoch == (args.epochs - 1):
            checkpoint = paddle.load(os.path.join(args.output_dir, args.task, 'checkpoint-best.pth'), map_location='cpu')
            model_without_ddp.set_state_dict(checkpoint['model'], strict=False)
            model.to(device)
            print("Test with the best model, epoch = %d:" % checkpoint['epoch'])
            test_stats, auc_roc = evaluate(data_loader_test, model, device, args, -1, mode='test',
                                           num_class=args.nb_classes, log_writer=None)

        if log_writer is not None:
            log_writer.add_scalar('loss/val', val_stats['loss'], epoch)

        # 确保所有值都是可序列化的
        train_stats_serializable = {f'train_{k}': v.item() if hasattr(v, 'item') else v for k, v in train_stats.items()}
        val_stats_serializable = {f'val_{k}': v.item() if hasattr(v, 'item') else v for k, v in val_stats.items()}
        
        log_stats = {**train_stats_serializable,
                     **val_stats_serializable,
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            # 暂时注释掉JSON日志记录，避免序列化问题
            # with open(os.path.join(args.output_dir, args.task, "log.txt"), mode="a", encoding="utf-8") as f:
            #     f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    criterion = paddle.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, criterion)


