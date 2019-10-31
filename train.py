import click
from models.i3d import I3D 
import torch
from data.dataset import Kinetics400


@click.command()
@click.option('-l', '--learning_rate', 'Initial Learning Rate', default=0.001)
@click.option('-m', '--momentum', "Momentum for SGD", default=0.9)
@click.option('-b', '--batch_size', 'Batch Size', default=8)
@click.option('-e', '--epochs', 'Number of Epochs to Train the Model for', default=300)
@click.option('-s', '--subsample', 'Subsample every N frames')
@click.option('-a', '--architecture', 'Architecture to Use', default='i3d')
@click.option('-d', '--dataset', 'Dataset to use', default='kinetics-400')
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
        train_dataset = Kinetics400('./data/400/kinetics_400_train.json')
        val_dataset = Kinetics400('./data/400/kinetics_400_validate.json')

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
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def val_epoch(model, device, val_loader, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += torch.nn.functional.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    print('\nVal: Average Loss: {:.4f}, Accuracy: {}/{}\n'.format(
        val_loss, correct, len(val_loader.dataset)
    ))


if __name__ == '__main__':
    train()