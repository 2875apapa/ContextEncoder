import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from datasets import *
from models import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import keyboard

# Setup device
cuda = (True if torch.cuda.is_available() else False)
if cuda:
    print("Use Cuda")
else:
    print("Use CPU")

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20)                             #numero di epoche training
parser.add_argument("--batch_size", type=int, default=4)                            #dimensione batches
parser.add_argument("--dataset_name", type=str, default="img_align_celeba")         #nome_dataset
parser.add_argument("--lr", type=float, default=0.0002)                             #learning reate
parser.add_argument("--b1", type=float, default=0.5)                                #b1
parser.add_argument("--b2", type=float, default=0.999)                              #b2
parser.add_argument("--n_cpu", type=int, default=12)                                #threads cpu in fase di training
parser.add_argument("--latent_dim", type=int, default=100)                          #latent space
parser.add_argument("--img_size", type=int, default=128)                            #dim immagine
parser.add_argument("--mask_size", type=int, default=64)                            #dim maschera
parser.add_argument("--channels", type=int, default=3)                              #channels
parser.add_argument("--sample_interval", type=int, default=500)                     #sample interval
opt = parser.parse_args()
print(opt)


# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
test_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def save_sample(batches_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)


#******************** Training**************************

if __name__ == '__main__':

    try:

        # Initialize history
        history_g_loss = []     #Generator Loss
        history_g_adv = []      #Generator Adversarial
        history_g_pixel = []    #Generator L2Loss
        history_d_loss = []     #Discriminator Loss


        for epoch in range(opt.n_epochs):

            sum_g_loss = 0
            sum_g_adv = 0
            sum_g_pixel = 0
            sum_d_loss = 0

            for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

                # Configure input
                imgs = Variable(imgs.type(Tensor))
                masked_imgs = Variable(masked_imgs.type(Tensor))
                masked_parts = Variable(masked_parts.type(Tensor))

                #  Train Generator

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_parts = generator(masked_imgs)

                # Adversarial and pixelwise loss
                g_adv = adversarial_loss(discriminator(gen_parts), valid)
                g_pixel = pixelwise_loss(gen_parts, masked_parts)

                # Total loss
                g_loss = 0.001 * g_adv + 0.999 * g_pixel

                g_loss.backward()
                optimizer_G.step()

                #  Train Discriminator

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(masked_parts), valid)
                fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)


                d_loss.backward()
                optimizer_D.step()

                #update history
                sum_g_loss += g_loss.item()
                sum_g_adv += g_adv.item()
                sum_g_pixel += g_pixel.item()
                sum_d_loss += d_loss.item()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                    % (epoch+1, opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
                )

                # Generate sample at sample interval
                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    save_sample(batches_done)

            # Compute epoch loss/accuracy
            epoch_g_loss = sum_g_loss/len(dataloader)
            epoch_g_adv = sum_g_adv/len(dataloader)
            epoch_g_pixel = sum_g_pixel/len(dataloader)
            epoch_d_loss = sum_d_loss/len(dataloader)

            # Update history
            history_g_loss.append(epoch_g_loss)
            history_g_adv.append(epoch_g_adv)
            history_g_pixel.append(epoch_g_pixel)
            history_d_loss.append(epoch_d_loss)
    
    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        # Plot Generator
        x = np.arange(1, opt.n_epochs+1, 1, dtype=int)
        plt.title("Generator Loss")
        plt.xlabel("Loss")
        plt.ylabel("Epoch")
        plt.plot(x,history_g_loss, label="Generator Loss")
        namepath="generator.png"
        plt.savefig(namepath)
        plt.close()
        np.savetxt('generator.txt', history_g_loss)


        # Plot Generator Adversarial
        x = np.arange(1, opt.n_epochs+1, 1, dtype=int)
        plt.title("Adversarial Loss")
        plt.xlabel("Adv Loss")
        plt.ylabel("Epoch")
        plt.plot(x,history_g_adv, label="Generator Adversarial")
        namepath="adversarial.png"
        plt.savefig(namepath)
        plt.close()
        np.savetxt('adversarial.txt', history_g_adv)


        #Plot Generator L2Loss
        x = np.arange(1, opt.n_epochs+1, 1, dtype=int)
        plt.title("L2Loss")
        plt.xlabel("L2")
        plt.ylabel("Epoch")
        plt.plot(x,history_g_pixel, label="Generator L2Loss")
        namepath="l2loss.png"
        plt.savefig(namepath)
        plt.close()
        np.savetxt('l2loss.txt', history_g_pixel)


        # Plot Discriminator
        x = np.arange(1, opt.n_epochs+1, 1, dtype=int)
        plt.title("Discriminator")
        plt.xlabel("Loss")
        plt.ylabel("Epoch")
        plt.plot(x,history_d_loss, label="Discriminator Loss")        
        namepath="discriminator.png"
        plt.savefig(namepath)
        plt.close()
        np.savetxt('discriminator.txt', history_d_loss)