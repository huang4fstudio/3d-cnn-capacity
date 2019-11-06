import click
import os
from models.i3d import I3D
import torch
from data.dataset import ActivityRecognitionDataset

from apex import amp
from apex.parallel import DistributedDataParallel

import wandb


@click.command()
@click.option('-l', '--learning_rate', help='Initial Learning Rate', default=0.0001)
@click.option('-m', '--momentum', help="Momentum for SGD", default=0.9)
@click.option('-b', '--batch_size', help='Batch Size', default=8)
@click.option('-e', '--epochs', help='Number of Epochs to Train the Model for', default=300)
@click.option('-s', '--subsample', help='Subsample every N frames')
@click.option('-a', '--architecture', help='Architecture to Use', default='i3d')
@click.option('-d', '--dataset', help='Dataset to use', default='ucf101')
@click.option('-r', '--restore', help='Checkpoint file', default=None)
@click.option('--distributed', help='use distributed training', is_flag=True, default=False)
@click.option('--local_rank')
# Wandb Project
@click.option('--dryrun', default=False, is_flag=True, help='Run the model as a dryrun')
@click.option('--wandb-project', default='video-captioning', help='The W&B Project to use')
@click.option('--wandb-tags', default=None, help='Comma separated list of tags to use for the run')
def train(**kwargs):

    if kwargs['dryrun']:
        os.environ['WANDB_MODE'] = 'dryrun'

    # Distributed training initialization
    if kwargs['distributed']:
        torch.cuda.set_device(int(kwargs['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(42)
    is_master_rank = not kwargs['distributed'] or (kwargs['distributed'] and int(kwargs['local_rank']) == 0)

    if is_master_rank:
        print('Initializing models/datasets')
        init_tags = kwargs['wandb_tags'].split(',') if kwargs['wandb_tags'] else []
        wandb.init(project='gerald', tags=init_tags)

    # Initialization
    model = None
    num_output_classes = -1

    lr = kwargs['learning_rate']
    architecture = kwargs['architecture']
    dataset_name = kwargs['dataset']
    batch_size = kwargs['batch_size']
    momentum = kwargs['momentum']
    epochs = kwargs['epochs']


    if dataset_name == 'kinetics-400':
        pass
        # num_output_classes = 400
        # train_dataset = Kinetics400('./data/400/kinetics_400_train.json')
        # val_dataset = Kinetics400('./data/400/kinetics_400_validate.json')
    elif dataset_name == 'ucf101':
        num_output_classes = 101
        train_dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_train.json', '/data/ucf101/downsampled/')
        val_dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_val.json', '/data/ucf101/downsampled/')
    else:
        raise NotImplementedError('This dataset is currently not supported!')

    if architecture == 'i3d':
        model = I3D(3, num_output_classes).cuda()
    else:
        raise NotImplementedError('This model is currently not supported!')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=1, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Apex AMP distributed code
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2') # Let's keep it at O2 for now

    # Distributed reduction
    if kwargs['distributed']:
        model = DistributedDataParallel(model)

    # Restore
    if kwargs['restore'] is not None:
        checkpoint = torch.load(kwargs['restore'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])

    if is_master_rank:
        print('Training...')

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch, is_master_rank)
        val_epoch(model, val_loader, epoch, batch_size, is_master_rank)

        # Save the model
        if is_master_rank:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
            }
            torch.save(checkpoint, 'amp_checkpoint.pt')


def train_epoch(model, train_loader, optimizer, epoch, is_master_rank):
    model.train()
    for batch_idx, example in enumerate(train_loader):
        data = example['video']
        target = example['class']
        data, target = data.cuda(), target.cuda()
        data = torch.transpose(data, 1, 4)
        data = torch.transpose(data, 2, 4)

        optimizer.zero_grad()

        # Compute the forward pass
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)

        # Loss stepping
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        if batch_idx % 20 == 0 and is_master_rank:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss, current batch: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({
                'Training Loss': loss.item(),
            })


def val_epoch(model, val_loader, epoch, batch_size, is_master_rank):
    model.eval()
    val_loss = 0
    correct = 0

    total_examples = 0
    with torch.no_grad():
        for idx, example in enumerate(val_loader):
            data = example['video']
            target = example['class']
            data, target = data.cuda(), target.cuda()
            data = torch.transpose(data, 1, 4)
            data = torch.transpose(data, 2, 4)
            output = model(data)
            val_loss += torch.nn.functional.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_examples += target.size()[0]
    val_loss /= len(val_loader)
    if is_master_rank:
        print('\nVal: Average Loss, per batch: {:.4f}, Accuracy: {}/{}\n'.format(
        val_loss, correct, len(val_loader) * batch_size
    ))

        print('Debug: Total Images: {} == {}'.format(total_examples, len(val_loader) * batch_size))
        wandb.log({
                'Validation Loss': loss.item(),
                'Validation Accuracy': correct/len(val_loader),
            })


if __name__ == '__main__':
    train()
