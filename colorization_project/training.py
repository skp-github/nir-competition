from models import *
import torch
from torch.utils.data import DataLoader
from datasets import NIRDataset


if __name__ == "__main__":
    lambda_recon = 200

    n_epochs = 1

    display_step = 500
    batch_size = 512
    lr = 0.002

    # TODO: Add your code to create the datasets here:
    dataset = NIRDataset()

    # TODO: Create the construct the model:
    model = Pix2Pix(in_channels=1,out_channels=3)
    model.compile("adam")

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        dataset_counter = 0 
        for i, data in enumerate(trainloader, 0):
            dataset_counter +=1 
            loss = model.train_step(data, i % 100 == 0)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
            if dataset_counter > 10 :
                break 

            if i % 100 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')

    torch.save(model.gen, 'colorizer.pt')
    torch.save(model.patch_gan, 'disc.pt')
