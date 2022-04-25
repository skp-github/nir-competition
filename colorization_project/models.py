import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import *
from gradsflow import Model

import matplotlib.pyplot as plt


DEVICE = 'cpu'
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.requires_grad_():
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d) and m.requires_grad_():
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def display_progress(cond, fake, figsize=(20, 10)):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(cond)
    ax[1].imshow(fake)
    plt.show()


class Colorizer:
    def __init__(self, model_name='model.pt', device='cuda'):
        # TODO: load your trained model with torch.load here:
        self.model = None
        pass

    def __call__(self, frame):
        img_out = None
        with torch.no_grad():
            # TODO: prepare the frame and call the trained model here:
            pass

        return img_out


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding, padding_mode='reflect')

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UpSampleConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True, dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 32 x 32
            UpSampleConv(1024, 256),  # bs x 256 x 64 x 64
            UpSampleConv(512, 64),  # bs x 64 x 256 x 256
        ]
        self.decoder_channels = [512, 512, 512, 512, 512, 256, 128, 64, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            m, n = skip.shape[-2:]

            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        x = self.final_conv(x)
        return self.tanh(x)


class PatchGAN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.d5 = DownSampleConv(512, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        x4 = self.d5(x3)
        xn = self.final(x4)
        return xn


class Pix2Pix(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, learning_rate=0.0002, lambda_recon=200, display_step=25,
                 model_names=None):
        super().__init__()

        self.display_step = display_step
        self.gen = Generator(in_channels, out_channels)
        self.patch_gan = PatchGAN(4)
        self.lambda_recon = lambda_recon
        self.display_step = display_step
        self.learning_rate = learning_rate

        # intializing weights
        if model_names is not None:
            self.gen = torch.load(model_names[0])
            self.patch_gan = torch.load(model_names[1])
        else:
            self.gen = self.gen.apply(_weights_init)


        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

        self.learner = [self.gen, self.patch_gan]

        for l in self.learner:
            l.to(DEVICE)


    def compile(self, optimizer="adam", learning_rate=3e-4):
        optimizer_fn = torch.optim.Adam # self._get_optimizer(optimizer)
        self.gen_optimizer = optimizer_fn(self.learner[0].parameters(), lr=learning_rate)
        self.disc_optimizer = optimizer_fn(self.learner[1].parameters(), lr=learning_rate)
        self.optimizer = optimizer_fn(self.learner[1].parameters(), lr=learning_rate)

        #self.disc_optimizer = self.prepare_optimizer(self.disc_optimizer)
        #self.gen_optimizer = self.prepare_optimizer(self.gen_optimizer)

        self._compiled = True

    @staticmethod
    def _batch_detect(net, img_batch, device):
        """
        Inputs:
            - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
        """

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        batch_size = img_batch.size(0)
        img_batch = img_batch.to(device, dtype=torch.float32)

        # img_batch = img_batch.flip(-3)  # RGB to BGR
        img_batch = img_batch - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

        # with torch.no_grad():
        olist = net(img_batch)  # patched uint8_t overflow error

        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)

        olist = [olist[i * 2] for i in range(len(olist) // 2)]
        # score = torch.stack(olist)
        # return score
        return olist[-4]

    def _gen_step(self, conditioned_images, rgb_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)

        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        recon_loss = self.recon_criterion(fake_images, rgb_images)
        lambda_recon = self.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, conditioned_images, real_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_logits = self.patch_gan(fake_images, conditioned_images)

        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def train_step(self, batch, display=False):
        condition, real = batch

        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

        # discriminator is at self.learner[1]
        disc_loss = self._disc_step(condition, real)
        #self.tracker.track_loss("discriminator/loss", disc_loss)
        # self.backward(disc_loss)
        disc_loss.backward()
        self.disc_optimizer.step()

        # generator is at self.learner[0]
        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()
        gen_loss = self._gen_step(condition, real)
        #self.tracker.track_loss("generator/loss", gen_loss)
        gen_loss.backward()
        self.gen_optimizer.step()

        loss = (disc_loss + gen_loss) / 2

        if display:
            fake = self.gen(condition).detach()
            display_progress(condition[0], fake[0])

        return loss

    def eval(self):
        for l in self.learner:
            l.eval()

    def train(self):
        for l in self.learner:
            l.train()