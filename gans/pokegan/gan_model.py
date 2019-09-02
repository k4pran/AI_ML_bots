import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from src import *
from src.image_transformation import get_dataset, get_dataloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class PokeGenerator(nn.Module):

    def __init__(self):
        super(PokeGenerator, self).__init__()
        self.gen_net = nn.Sequential(
            nn.ConvTranspose2d(GEN_LATENT_INPUT, GEN_FEATURE_MAPS * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FEATURE_MAPS * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(GEN_FEATURE_MAPS * 8, GEN_FEATURE_MAPS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURE_MAPS * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(GEN_FEATURE_MAPS * 4, GEN_FEATURE_MAPS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURE_MAPS * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(GEN_FEATURE_MAPS * 2, GEN_FEATURE_MAPS, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURE_MAPS),
            nn.ReLU(True),

            nn.ConvTranspose2d(GEN_FEATURE_MAPS, COLOR_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.gen_net(input)


class PokeDiscriminator(nn.Module):

    def __init__(self):
        super(PokeDiscriminator, self).__init__()

        self.disc_net = nn.Sequential(
            nn.Conv2d(COLOR_CHANNELS, DISC_FEATURE_MAPS, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(DISC_FEATURE_MAPS, DISC_FEATURE_MAPS * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURE_MAPS * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(DISC_FEATURE_MAPS * 2, DISC_FEATURE_MAPS * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURE_MAPS * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(DISC_FEATURE_MAPS * 4, DISC_FEATURE_MAPS * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURE_MAPS * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(DISC_FEATURE_MAPS * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.disc_net(input)

device = torch.device("cuda:0" if (torch.cuda.is_available() and NUMBER_GPUS > 0) else "cpu")

pokegen_net = PokeGenerator()
pokegen_net.apply(weights_init)

pokedisc_net = PokeDiscriminator()
pokedisc_net.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, GEN_LATENT_INPUT, 1, 1, device=device)

real_label = 1
fake_label = 0

gen_optimizer = optim.Adam(pokegen_net.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
disc_optimizer = optim.Adam(pokedisc_net.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))


dataset = get_dataset(transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

dataloader = get_dataloader(dataset)

generated_images = []
gen_losses = []
disc_losses = []
iterations = 0

print("Beginning training...")
for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader, 0):

        # Train discriminator
        pokedisc_net.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        output = pokedisc_net(real_cpu).view(-1)
        err_disc_real = criterion(output, label)
        err_disc_real.backward()
        disc_pred = output.mean().item()

        noise = torch.randn(b_size, GEN_LATENT_INPUT, 1, 1, device=device)
        fake = pokegen_net(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = pokedisc_net(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = err_disc_real + errD_fake
        # Update D
        disc_optimizer.step()

        # Train generator
        pokegen_net.zero_grad()
        label.fill_(real_label)
        output = pokedisc_net(fake).view(-1)
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()
        gen_optimizer.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(dataloader),
                     errD.item(), errG.item(), disc_pred, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        gen_losses.append(errG.item())
        disc_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iterations % 500 == 0) or ((epoch == EPOCHS - 1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = pokegen_net(fixed_noise).detach().cpu()
            generated_images.append(vutils.make_grid(fake, padding=2, normalize=True))

        iterations += 1

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in generated_images]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(generated_images[-1], (1, 2, 0)))
plt.show()