import logging
import os
import sys
import warnings
import numpy as np
import torch
from torch import distributed as dist

LOG = logging.getLogger(__name__)


def boolean_string(s):
    """Enable {'False', 'True'} string args in command line"""
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate_cyclic(optimizer, current_epoch, start_epoch,
                                swa_freqent=5, lr_max=4e-5, lr_min=2e-5):
    epoch = current_epoch - start_epoch

    lr = lr_max - (lr_max - lr_min) / (swa_freqent - 1) * (
            epoch - epoch // swa_freqent * swa_freqent)
    lr = round(lr, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_model(path, epoch, train_loss, model, optimizer=None, amp_state=None):
    from apex.parallel import DistributedDataParallel
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        state_dict = model.module.state_dict()  # remove prefix 'module.'
    else:
        state_dict = model.state_dict()
    print(f'Saving {model.__class__.__name__} state dict...')

    data = {'epoch': epoch,
            'train_loss': train_loss,
            'model_state_dict': state_dict}
    if optimizer is not None:
        print(f'Saving {optimizer.__class__.__name__} state dict...')
        data['optimizer_state_dict'] = optimizer.state_dict()
    if amp_state is not None:
        print(f'Apex is used, saving all loss_scalers and their corresponding unskipped steps...')
        data['amp'] = amp_state
    torch.save(data, path)
    print(f'Checkpoint has been saved at {path}')


def load_model(model, ckpt_path, *, optimizer=None, drop_layers=True, drop_name='offset_convs',
               resume_optimizer=True, optimizer2cuda=True, load_amp=False):
    """
    Load pre-trained model and optimizer checkpoint.
    Args:
        model:
        ckpt_path:
        optimizer:
        drop_layers (bool): drop pre-trained params of the output layers, etc
        drop_name: drop layers with this string in names
        resume_optimizer:
        optimizer2cuda (bool): move optimizer statues to cuda
        load_amp (bool): load the amp state including loss_scalers
            and their corresponding unskipped steps
    """

    start_epoch = 0
    start_loss = float('inf')
    if not os.path.isfile(ckpt_path):
        print(f'WARNING!! ##### Current checkpoint file {ckpt_path} DOSE NOT exist!!#####')
        warnings.warn("No pre-trained parameters are loaded!"
                      " Please make sure you initialize the model randomly!")
        user_choice = input("Are you sure want to continue with randomly model (y/n):\n")
        if user_choice in ('y', 'Y'):
            # return without loading
            load_amp = False
            return model, optimizer, start_epoch, start_loss, load_amp
        else:
            sys.exit(0)

    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if 'model' in checkpoint.keys():
        state_dict_ = checkpoint['model']  # type: dict
        LOG.info('Loading the pre-trained model that is pre-trained by MAE on AffectNet dataset.')
        assert (ckpt_path.__contains__('vit-base-checkpoint-300.pth') or
                ckpt_path.__contains__('vit-tiny-checkpoint-299.pth') or
                ckpt_path.__contains__('vit-small-checkpoint-299.pth') or
                ckpt_path.__contains__('mae_pretrain_vit_base.pth')
                )
        print(f'Loading MAE pre-trained model {ckpt_path}')
    elif 'model_state_dict' in checkpoint.keys():
        start_epoch = checkpoint['epoch'] + 1
        start_loss = checkpoint['train_loss']
        state_dict_ = checkpoint['model_state_dict']  # type: dict

        LOG.info('Loading pre-trained model %s, checkpoint at epoch %d', ckpt_path,
                 checkpoint['epoch'])
    else:
        state_dict_ = checkpoint
        start_loss = float('inf')
        start_epoch = 0
        print(f'Loading MAEGAN pre-trained model {ckpt_path}')

    if load_amp and 'amp' in checkpoint.keys():
        LOG.info('Found saved amp state including loss_scalers and their corresponding '
                 'unskipped steps from checkpoint %s at epoch %d', ckpt_path, start_epoch)
        amp = checkpoint['amp']
    else:
        print(f'No OLD amp state is detected from current checkpoint {ckpt_path} '
              f'or you do not load amp state')
        amp = False

    from collections import OrderedDict
    state_dict = OrderedDict()  # loaded pre-trained model weight

    # convert parallel/distributed model to single model
    for k, v in state_dict_.items():  # Fixme: keep consistent with our model
        if (drop_name in k or 'some_example_convs' in k) and drop_layers:  #
            continue
        if k.startswith('module') and not k.startswith('module_list'):
            name = k[7:]  # remove prefix 'module.'
            # name = 'module.' + k  # add prefix 'module.'
            state_dict[name] = v
        else:
            name = k
            state_dict[name] = v
    model_state_dict = model.state_dict()  # newly built model

    # check loaded parameters and created model parameters
    msg1 = 'If you see this, your model does not fully load the ' + \
           'pre-trained weight. Please make sure ' + \
           'you have correctly built the model layers or the weight shapes.'
    msg2 = 'If you see this, your model has more parameters than the ' + \
           'pre-trained weight. Please make sure ' + \
           'you have correctly specified more layers.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                LOG.debug(
                    'Skip loading pre-trained parameter %s, current model '
                    'required shape %s, loaded shape %s. %s',
                    k, model_state_dict[k].shape, state_dict[k].shape, msg1)
                state_dict[k] = model_state_dict[k]  # fix badly mismatched params
        else:
            LOG.debug('Drop pre-trained parameter %s which current model dose '
                      'not have. %s', k, msg1)
    for k in model_state_dict:
        if not (k in state_dict):
            LOG.debug('No param %s in pre-trained model. %s', k, msg2)
            state_dict[k] = model_state_dict[k]  # append missing params to rescue
    message = model.load_state_dict(state_dict, strict=False)
    # print(message)
    print(f'Network {model.__class__.__name__} weights have been resumed from checkpoint: {ckpt_path}')

    # resume optimizer parameters
    if optimizer is not None and resume_optimizer:
        if 'optimizer_state_dict' in checkpoint:
            LOG.debug('Resume the optimizer.')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Here, we must convert the resumed state data of optimizer to gpu.
            # In this project, we use map_location to map the state tensors to cpu.
            # In the training process, we need cuda version of state tensors,
            # so we have to convert them to gpu.
            if torch.cuda.is_available() and optimizer2cuda:
                LOG.debug('Move the optimizer states into GPU.')
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            # param_group['lr'] will be instead set in a separate fun: adjust_learning_rate()
            print('Optimizer {} has been resumed from the checkpoint at epoch {}.'
                  .format(optimizer.__class__.__name__, start_epoch - 1))
        elif optimizer is not None:
            print('Optimizer {} is NOT resumed, although the checkpoint exists.'.format(optimizer.__class__.__name__))
        else:
            print('Optimizer is {}.'.format(optimizer))
    return model, optimizer, start_epoch, start_loss, amp


def reduce_tensor(tensor, world_size):
    # Reduces the tensor data on GPUs across all machines
    # If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1'), here is cuda:  cuda:1
    # tensor(340.1970, device='cuda:0'), here is cuda:  cuda:0
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


