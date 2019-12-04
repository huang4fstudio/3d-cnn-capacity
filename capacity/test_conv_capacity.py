import torch
import numpy as np
import click
import tqdm

from alexandria.layers.util import Flatten

@click.command()
def main():

    min_pts = 0
    current_pts = 25000
    max_pts = 50000
    num_classes = 2
    batch_size = 256
    COUNT_LIMIT = 500

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
                    if best_acc > 0.99:
                        break
                else:
                    static_ct += 1
                    if static_ct > COUNT_LIMIT:
                        break
                pbar.set_description_str('L: {}, A: {}, S: {}'.format(
                    np.mean(epoch_acc), best_acc, static_ct
                ))
                pbar.update(1)


            if static_ct <= COUNT_LIMIT:
                min_pts = current_pts
            else:
                max_pts = current_pts

            if abs(max_pts-min_pts) < COUNT_LIMIT:
                break

            current_pts = (max_pts - min_pts) // 2 + min_pts

    print('Done!')
    print('Max: {}, Min: {}'.format(max_pts, min_pts))


if __name__ == '__main__':
    main()