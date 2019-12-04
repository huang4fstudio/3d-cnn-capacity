import torch
import numpy as np
import click
import tqdm

from alexandria.layers.util import Flatten

@click.command()
@click.option('--min-points', default=0, help='Min number of points in the search')
@click.option('--max-points', default=50000, help='Max number of points in the search')
@click.option('--n-classes', default=2, help='Number of classes in the problem')
@click.option('--batch-size', default=256, help='Model batch size')
@click.option('--term-count', default=500, help='How many epochs to run before giving up')
@click.option('--success-threshold', default=0.99, help='The threshold which is considered a success')
def main(**kwargs):

    min_pts = kwargs['min_points']
    max_pts = kwargs['max_points']
    current_pts = (max_pts - min_pts) // 2 + min_pts
    num_classes = kwargs['n_classes']
    batch_size = kwargs['batch_size']
    count_limit = kwargs['term_count']

    while True:
        num_points = current_pts
        print('Testing {} points...'.format(num_points))
        random_data = torch.tensor(np.random.rand(num_points, 3, 16, 16)).float()
        random_labels = torch.tensor(np.random.randint(0, num_classes, size=[num_points]))

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, (4,4), (2,2)),
            torch.nn.Sigmoid(),
            Flatten(),
            torch.nn.Linear(7*7*8, num_classes),
        )
        criterion = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.cuda()

        opt = torch.optim.Adam(model.parameters(), lr=3e-4)

        dataset = torch.utils.data.TensorDataset(random_data, random_labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)

        best_acc = 0
        static_ct = 0
        with tqdm.tqdm() as pbar:
            while True:
                epoch_losses = []
                epoch_acc = []
                for (data, labels) in data_loader:

                    if torch.cuda.is_available():
                        data = data.cuda()
                        labels = labels.cuda()

                    opt.zero_grad()
                    model_output = model(data)
                    loss = criterion(model_output, labels)
                    with torch.no_grad():
                        epoch_acc.append((model_output.argmax(dim=1) == labels).sum().item()/data.shape[0])
                    loss.backward()
                    opt.step()
                    epoch_losses.append(loss.item())


                if np.mean(epoch_acc) > best_acc:
                    best_acc = np.mean(epoch_acc)
                    static_ct = 0
                    if best_acc > kwargs['success_threshold']:
                        break
                else:
                    static_ct += 1
                    if static_ct > count_limit:
                        break
                pbar.set_description_str('L: {}, A: {}, S: {}'.format(
                    np.mean(epoch_acc), best_acc, static_ct
                ))
                pbar.update(1)


            if static_ct <= count_limit:
                min_pts = current_pts
            else:
                max_pts = current_pts

            if abs(max_pts-min_pts) < count_limit:
                break

            current_pts = (max_pts - min_pts) // 2 + min_pts

    print('Done!')
    print('Max: {}, Min: {}'.format(max_pts, min_pts))


if __name__ == '__main__':
    main()
