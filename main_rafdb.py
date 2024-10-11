"""Distributed training with Nvidia Apex;
Follow https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py"""
import argparse
import logging
import os
import sys
import time
import datetime

import torch
import torch.distributed as dist
import torchvision
from PIL import Image
import timm
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from data import RafDataset, DistributedSamplerWrapper, ConservativeImbalancedDatasetSampler


import logs
import utils
from utils.util import reduce_tensor
import utils.lr_decay as lrd
from models import vit

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    import apex.optimizers as apex_optim
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example.")

# os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2"
import warnings

warnings.filterwarnings('ignore')

LOG = logging.getLogger(__name__)


def train_cli():
    parser = argparse.ArgumentParser(
        # __doc__: current module's annotation (or module.a_function's annotation)
        description=__doc__,
        # --help text with default values, e.g., gaussian threshold (default: 0.1)
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    logs.cli(parser)

    parser.add_argument('--raf_path', type=str, default='data/images_labels/link2RAF-DB/basic', help='raf_dataset_path')
    parser.add_argument('--num_class', type=int, default=7, help='number of emotion class in the current dataset')
    parser.add_argument('--train_label_path', type=str, default='data/images_labels/link2RAF-DB/basic/EmoLabel'
                                                                '/train_label.txt',
                        help='label_path')
    parser.add_argument('--test_label_path', type=str, default='data/images_labels/link2RAF-DB/basic/EmoLabel'
                                                               '/test_label.txt',
                        help='label_path')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size per GPU process')
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224', help='model name from pretrained '
                                                                                       'checkpoint')
    parser.add_argument('--checkpoint-path', '-p',
                        default='checkpoint', help='folder path for checkpoint storage')
    parser.add_argument('--checkpoint-whole', default='checkpoint/ViT_118_epoch.pth', type=str,
                        help='the checkpoint path to the whole model')

    parser.add_argument('--resume', '-r', action='store_true', default=False,
                        help='resume from checkpoint')
    parser.add_argument('--drop-optim-state', dest='resume_optimizer', action='store_false',
                        default=True,
                        help='do not resume the optimizer from checkpoint.')
    parser.add_argument('--drop-amp-state', dest='load_amp', action='store_false',
                        default=True,
                        help='do not resume the apex apm statue from checkpoint.')
    parser.add_argument('--drop-layers', action='store_true', default=False,
                        help='drop some layers described in utils/util.py:173')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--warmup', action='store_true', default=False,
                        help='using warm-up learning rate')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',  # 没有动作时默认时False
                        help='evaluate model on validation set')

    # Apex configuration
    group = parser.add_argument_group('apex configuration')
    group.add_argument("--local_rank", default=0, type=int)
    # full precision O0  # mixture precision O1 # half precision O2
    group.add_argument('--opt-level', type=str, default='O1')
    group.add_argument('--no-sync-bn', dest='sync_bn', action='store_false',
                       default=True,
                       help='enabling apex sync BN.')
    group.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    group.add_argument('--loss-scale', type=str, default=None)  # '1.0'
    group.add_argument('--channels-last', default=False, action='store_true',
                       help='channel last may lead to 22% speed up')  # not implemented yet
    group.add_argument('--print-freq', '-f', default=100, type=int, metavar='N',
                       help='print frequency (default: 10)')

    # Optimizer configuration
    group = parser.add_argument_group('optimizer configuration')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam'])
    group.add_argument('--loss-fn', type=str, default='CE',
                       choices=['CE', 'smoothCE'])
    group.add_argument('--mixup', dest='use_mixup', action='store_true',
                       default=False,
                       help='training the model using MixUp augmentation')
    group.add_argument('--learning-rate', type=float, default=1e-4,
                       metavar='LR',
                       help='learning rate in 1 world size, '
                            'thus, the actual LR will learning_rate * world_size')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                       help='momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (default: 0.005? 1e-4)')
    parser.add_argument('--layer-decay', type=float, default=1,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    args = parser.parse_args()

    if args.logging_output is None:
        args.logging_output = default_output_file(args)
    return args

def prepare_dataset(args, prem_cfg):
    data_augment = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                                                               saturation=0.25, hue=0.25),
                                            # 仿射变换对训练准确率有显著影响，但是不能太剧烈，以下设置效果就挺好
                                            torchvision.transforms.RandomAffine(degrees=0, translate=(.1, .1),
                                                                                scale=(1.0, 1.25),
                                                                                resample=Image.BILINEAR)
                                            ], p=0.5),

        torchvision.transforms.Resize((prem_cfg['input_size'][1], prem_cfg['input_size'][2])),  # original size: 224*224
        torchvision.transforms.ToTensor(),
        # mean=[0.485, 0.456, 0.406], IMAGENET_DEFAULT_MEAN
        # std=[0.229, 0.224, 0.225], IMAGENET_DEFAULT_STD
        torchvision.transforms.Normalize(mean=list(prem_cfg['mean']),
                                         std=list(prem_cfg['std'])),
        torchvision.transforms.RandomErasing(scale=(0.02, 0.25))  # beneficial to final accuracy
    ])
    data_transforms_val = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((prem_cfg['input_size'][1], prem_cfg['input_size'][2])), # original size: 224*224
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=list(prem_cfg['mean']),
                                         std=list(prem_cfg['std']))
    ])
    train_dataset = RafDataset(args, args.train_label_path, transform=data_augment)
    print('Training set size:', train_dataset.__len__())
    test_dataset = RafDataset(args, args.test_label_path, transform=data_transforms_val)
    print('Validation set size:', test_dataset.__len__())
    # concat_data = torch.utils.data.ConcatDataset([train_dataset, test_dataset])  # More training data for example
    return test_dataset, train_dataset


def main():
    global best_loss, args
    best_loss = float('inf')
    start_epoch = 0
    args = train_cli()
    logs.configure(args)

    print(f"\nopt_level = {args.opt_level}")
    print(f"keep_batchnorm_fp32 = {args.keep_batchnorm_fp32}")
    print(f"loss_scale = {args.loss_scale}")
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    print(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")

    if args.local_rank == 0:
        # build epoch recorder
        # os.makedirs(args.checkpoint_path, exist_ok=True)
        with open(os.path.join('./' + args.checkpoint_path, 'log'), 'a+') as f:
            f.write('\n\ncmd line: {} \n \targs dict: {}'.format(' '.join(sys.argv), vars(args)))
            f.flush()

    torch.backends.cudnn.benchmark = True
    # print(vars(args))
    args.pin_memory = False
    args.use_cuda = False
    if torch.cuda.is_available():
        args.use_cuda = True
        args.pin_memory = True
    else:
        raise ValueError(f'CUDA is available: {args.use_cuda}')

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    # FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
    # the 'WORLD_SIZE' environment variable will also be set automatically.
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()  # get the current process id
        print("World Size is :", args.world_size)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # #################################################################
    # ############## 全局取消证书验证，否则timm包下载权值失败 ###############
    # #################################################################
    import ssl
    # 当timm包使用urllib.urlopen打开一个 https链接时，会验证一次 SSL 证书
    ssl._create_default_https_context = ssl._create_unverified_context

    # ##################################################################
    # ################# Create the Vision Transformer #################
    # ##################################################################
    # A blog for vit in timm: https://zhuanlan.zhihu.com/p/350837279
    model = timm.create_model(args.model_name, pretrained=not args.resume, num_classes=args.num_class)
    print(f'Modified {args.model_name} classifier: {model.get_classifier()}')
    prem_cfg = model.default_cfg  # pre-trained models may have different preprocessing configs

    test_dataset, train_dataset = prepare_dataset(args, prem_cfg)

    if args.sync_bn:
        #  This should be done before model = DDP(model, delay_allreduce=True),
        #  because DDP needs to see the finalized model parameters
        import apex

        print("Using apex synced BN.")
        model = apex.parallel.convert_syncbn_model(model)

        # NOTICE! It should be called before constructing optimizer
        # if the module will live on GPU while being optimized.
    model.cuda()

    # # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )

    # if args.optimizer == 'sgd':
    #     # optimizer = apex_optim.FusedSGD(
    #     # filter(lambda p: p.requires_grad, model.parameters()),
    #     # lr=opt.learning_rate * args.world_size, momentum=0.9, weight_decay=5e-4)
    #     optimizer = torch.optim.SGD(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=args.learning_rate * args.world_size, momentum=args.momentum,
    #         weight_decay=args.weight_decay)
    # elif args.optimizer == 'adam':
    #     optimizer = apex_optim.FusedAdam(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=args.learning_rate * args.world_size,
    #         weight_decay=args.weight_decay)
    # else:
    #     raise Exception(f'optimizer {args.optimizer} is not supported')

    if args.optimizer == 'adam':
        optimizer = apex_optim.FusedAdam(
            param_groups,
            lr=args.learning_rate * args.world_size)
    else:
        raise Exception(f'optimizer {args.optimizer} is not supported')

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)  # Dynamic loss scaling is used by default.

    if args.resume:
        model, optimizer, start_epoch, best_loss, amp_state = utils.load_model(
            model, args.checkpoint_whole, optimizer=optimizer, resume_optimizer=args.resume_optimizer,
            drop_layers=False, optimizer2cuda=args.use_cuda, load_amp=args.load_amp)
        if amp_state:
            print(f'Amp state has been restored from the checkpoint at epoch {start_epoch - 1}.')
            amp.load_state_dict(amp_state)  # load amp_state after amp.initialize

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    if args.loss_fn == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_fn == 'smoothCE':
        criterion = LabelSmoothingCrossEntropy()

    mixup_fn = None
    if args.use_mixup:  # undo: remove mixup temporarily
        mixup_fn = Mixup(num_classes=args.num_class)
        # mixup_fn = Mixup(
        #     mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        #     prob=0.1, switch_prob=0.5, mode='batch',
        #     label_smoothing=0.1, num_classes=args.num_class)
        criterion = SoftTargetCrossEntropy()  # FIXME: 需要把test阶段的loss换回普通的CE吗

    #  ################# RAFDB之前的实验没有添加过平衡采样  #################
    train_sampler = None
    val_sampler = None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)  # concat_data
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    # train_sampler = ConservativeImbalancedDatasetSampler(train_dataset)  # 使用权值平衡采样后精度反而降低了
    # val_sampler = None
    #
    # if args.distributed:
    #     train_sampler = DistributedSamplerWrapper(train_sampler)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

        # 创建数据加载器，在训练和验证步骤中喂数据
    train_loader = torch.utils.data.DataLoader(train_dataset,  # concat_data
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory,
                                               sampler=train_sampler,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin_memory,
                                             sampler=val_sampler,
                                             drop_last=False)
    if args.evaluate:
        test(val_loader, val_sampler, model, criterion, optimizer, 0)
        return

    best_acc = 0
    best_epoch = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(train_loader, train_sampler, model, criterion, optimizer, epoch, mixup_fn=mixup_fn)
        test_acc = test(val_loader, val_sampler, model, criterion, optimizer, epoch)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        print(f'best acc {best_acc} @ epoch {best_epoch}')


def train(train_loader, train_sampler, model, criterion, optimizer, epoch, mixup_fn=None):
    print('\n ======================= Train phase, Epoch: {} ======================='.format(
        epoch))
    torch.cuda.empty_cache()
    model.train()
    # disturb and allocation data differently at each epcoh
    # train_sampler make each GPU process see 1/(world_size) training samples per epoch
    if args.distributed:
        #  calling the set_epoch method is needed to make shuffling work
        train_sampler.set_epoch(epoch)

    # adjust_learning_rate_cyclic(optimizer, epoch, start_epoch)  # start_epoch
    print(
        '\nLearning rate at this epoch is: %0.9f\n' % optimizer.param_groups[0][
            'lr'])

    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1_acc = utils.AverageMeter()
    end = time.time()

    for batch_idx, (images, labels, idxs) in enumerate(train_loader):
        # # ##############  Use fun of 'adjust learning rate' #####################
        adjust_learning_rate(args.learning_rate, args.world_size,
                                   optimizer, epoch, batch_idx, len(train_loader),
                                   use_warmup=args.warmup)
        LOG.debug('\nLearning rate at this batch is: %0.9f', optimizer.param_groups[0]['lr'])
        # # ##########################################################

        #  这允许异步 GPU 复制数据也就是说计算和数据传输可以同时进.
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        if args.use_mixup:  # undo: remove mixup temporarily
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()  # zero the gradient buff

        outputs = model(images)

        loss = criterion(outputs, labels)

        if loss.item() > 1e8:  # try to rescue the gradient explosion
            import warnings
            warnings.warn("\nOh My God! \nLoss is abnormal, drop this batch!")
            loss.zero_()

        LOG.info({
            'type': f'train-at-rank{args.local_rank}',
            'epoch': epoch,
            'batch': batch_idx,
            'loss': round(to_python_float(loss.detach()), 6),
        })

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # According to our experience, clip norm is easy to destroy the training after few epochs
        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)

        optimizer.step()

        if batch_idx % args.print_freq == 0:
            # print不应该多用 会触发allreduce，而这个操作比较费时，因此只计算training set上部分数据（1/preint_freq）的loss和metric
            if args.use_mixup:
                prec = 0
            else:
                prec = utils.accuracy(outputs.data, labels)[0]
            # print('***************', args.distributed)  # False. fixme bug, 但是在单个GPU上不影响
            if args.distributed:
                # We manually reduce and average the metrics across processes. In-place reduce tensor.
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                reduced_prec = reduce_tensor(prec, args.world_size)
            else:
                reduced_loss = loss.data
                if args.use_mixup:
                    reduced_prec = [0]
                else:
                    reduced_prec = prec.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
            top1_acc.update(to_python_float(reduced_prec), images.size(0))
            torch.cuda.synchronize()  # 因为所有GPU操作是异步的，应等待当前设备上所有流中的所有核心完成，测试的时间才正确
            # 注意计时循环在1/print_freq内部，end时间包括了整个print_freq时间
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:  # Print them in the Process 0
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Accuracy {top1_acc.val:.3f} ({top1_acc.avg:.3f})\t'.format(
                    epoch, batch_idx, len(train_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses,
                    top1_acc=top1_acc))

    # global best_loss
    # # DistributedSampler控制进入分布式环境的数据集以确保模型不是对同一个子数据集训练，以达到训练目标。
    #
    if args.local_rank == 0:
        # Write the log file each epoch.
        with open(os.path.join('./' + args.checkpoint_path, 'log'), 'a+') as f:
            # validation时写字符串后不要\n换行
            f.write('\nEpoch {}\ttrain_loss: {}\taccuracy:{}'.format(epoch, losses.avg, top1_acc.avg))
            f.flush()

        if losses.avg < float('inf'):  # < best_loss
            # Update the best_loss if the average loss drops
            best_loss = losses.avg  # todo: modify best_loss to best_AP

            save_path = './' + args.checkpoint_path + '/ViTRafDb_' + str(epoch) + '_epoch.pth'
            utils.save_model(save_path, epoch, best_loss,
                             model, optimizer, amp_state=amp.state_dict())


def test(val_loader, val_sampler, model, criterion, optimizer, epoch):
    print('\n ======================= Test phase, Epoch: {} ======================='.format(epoch))
    model.eval()
    # DistributedSampler 中记录目前的 epoch 数， 因为采样器是根据 epoch 来决定如何打乱分配数据进各个进程
    # if args.distributed:
    #     val_sampler.set_epoch(epoch)  # 验证集太小，不够4个划分
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1_acc = utils.AverageMeter()
    end = time.time()

    for batch_idx, (images, labels, idxs) in enumerate(val_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda()

        with torch.no_grad():
            outputs = model(images)

            # print the prediction score
            # probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # _, preds = torch.topk(probabilities, 1)
            # print(preds)

            loss = criterion(outputs, labels)

        LOG.info({
            'type': f'validate-at-rank{args.local_rank}',
            'epoch': epoch,
            'batch': batch_idx,
            'loss': round(to_python_float(loss.detach()), 6),
        })

        # measure accuracy and record loss
        prec = utils.accuracy(outputs.data, labels, topk=(1,))[0]

        if args.distributed:
            # We manually reduce and average the metrics across processes. In-place reduce tensor.
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_prec = reduce_tensor(prec, args.world_size)
        else:
            reduced_loss = loss.data
            reduced_prec = prec.data
        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
        top1_acc.update(to_python_float(reduced_prec), images.size(0))
        torch.cuda.synchronize()
        batch_time.update((time.time() - end))
        end = time.time()

        if args.local_rank == 0 and batch_idx % args.print_freq == 0:  # Print them in the Process 0
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                  'Accuracy {top1_acc.val:.3f} ({top1_acc.avg:.3f})\t'.format(
                epoch, batch_idx, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time,
                loss=losses,
                top1_acc=top1_acc))

    if args.local_rank == 0:  # Print them in the Process 0
        # Write the log file each epoch.
        with open(os.path.join('./' + args.checkpoint_path, 'log'), 'a+') as f:
            f.write('\tval_loss: {}\taccuracy: {}'.format(losses.avg, top1_acc.avg))  # validation时不要\n换行
            f.flush()

    print(f'>>>>>>>>>>>>>>>>>>> Evaluation on the test dataset: {top1_acc.avg} % Accuracy <<<<<<<<<<<<<<<<<<<<<<<')
    return top1_acc.avg


def default_output_file(args):
    out = 'logs/outputs/{}-{}'.format('rafdb', ''.join(args.model_name))
    now = datetime.datetime.now().strftime('%Y-%m%d-%H%M%S')
    out += '-{}.pkl'.format(now)

    return out


def adjust_learning_rate(learning_rate, world_size, optimizer,
                         epoch, step, len_epoch, use_warmup=False):
    """
    Scale the LR by world size: lr = learning_rate * world_size
    """
    # factor = epoch // 15
    #
    # lr = learning_rate * world_size * (0.2 ** factor)

    lr = learning_rate * world_size

    """Warmup the learning rate"""
    if use_warmup:
        if epoch < 5:
            # print('=============>  Using warm-up learning rate....')
            lr = lr * float(1 + step + epoch * len_epoch) / (
                    5. * len_epoch)  # len_epoch=len(train_loader)

    if 5 <= epoch < 35:
        lr = learning_rate * world_size

    if 35 <= epoch < 65:
        lr = 0.1 * learning_rate * world_size

    if 65 <= epoch < 95:
        lr = 0.01 * learning_rate * world_size

    if 95 <= epoch:
        lr = 0.001 * learning_rate * world_size

    # if 92 <= epoch < 105:
    #     lr = 0.1 * learning_rate * world_size
    #
    # if 105 <= epoch < 110:
    #     lr = 0.01 * learning_rate * world_size

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
