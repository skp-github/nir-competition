from models import *
import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    lambda_recon = 200

    n_epochs = 1

    display_step = 500
    batch_size = 512
    lr = 0.002

    # TODO: Add your code to create the datasets here:
    dataset = None

    # TODO: Create the construct the model:
    model = Pix2Pix()
    model.compile("adam")

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            loss = model.train_step(data, i % 100 == 0)

            if i % 100 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')

    torch.save(model.gen, 'colorizer.pt')
    torch.save(model.patch_gan, 'disc.pt')
