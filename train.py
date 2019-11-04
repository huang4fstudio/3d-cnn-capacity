import click
from models.i3d import I3D 
import torch
from data.dataset import ActivityRecognitionDataset


@click.command()
@click.option('-l', '--learning_rate', help='Initial Learning Rate', default=0.0001)
@click.option('-m', '--momentum', help="Momentum for SGD", default=0.9)
@click.option('-b', '--batch_size', help='Batch Size', default=8)
@click.option('-e', '--epochs', help='Number of Epochs to Train the Model for', default=300)
@click.option('-s', '--subsample', help='Subsample every N frames')
@click.option('-a', '--architecture', help='Architecture to Use', default='i3d')
@click.option('-d', '--dataset', help='Dataset to use', default='ucf101')
def train(**kwargs):
    model = None
    num_output_classes = -1
 
    lr = kwargs['learning_rate']
    architecture = kwargs['architecture']
    dataset_name = kwargs['dataset']
    batch_size = kwargs['batch_size']
    momentum = kwargs['momentum']
    epochs = kwargs['epochs']

    
    if dataset_name == 'kinetics-400':
        num_output_classes = 400
        # train_dataset = Kinetics400('./data/400/kinetics_400_train.json')
        # val_dataset = Kinetics400('./data/400/kinetics_400_validate.json')
    
    if dataset_name == 'ucf101':
        num_output_classes = 101
        train_dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_train.json', '/data/ucf101/downsampled/')
        val_dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_val.json', '/data/ucf101/downsampled/')

    if num_output_classes == -1:
        raise NotImplementedError('This dataset is currently not supported!')

    device = torch.device("cuda")

    if architecture == 'i3d':
        model = I3D(3, num_output_classes).to(device)

    if model is None:
        raise NotImplementedError('This model is currently not supported!')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(1, epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)
        val_epoch(model, device, val_loader, epoch)


def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, example in enumerate(train_loader):
        data = example['video']
        target = example['class']
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        data = torch.transpose(data, 1, 4)
        data = torch.transpose(data, 2, 4)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def val_epoch(model, device, val_loader, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for example in val_loader:
            data = example['video']
            target = example['class']
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            data = torch.transpose(data, 1, 4)
            data = torch.transpose(data, 2, 4)
            output = model(data)
            val_loss += torch.nn.functional.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    print('\nVal: Average Loss: {:.4f}, Accuracy: {}/{}\n'.format(
        val_loss, correct, len(val_loader.dataset)
    ))


if __name__ == '__main__':
    train()
