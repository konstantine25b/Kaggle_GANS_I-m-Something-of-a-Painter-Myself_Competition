import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

from models.generator.resnet_gan import ResNetGenerator
from models.discriminator.patch_gan import PatchDiscriminator
from utils.dataset import ImageDataset, get_transforms
from utils.helpers import ReplayBuffer, weights_init_normal

def train(args):
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize networks
    # G_Monet: Photo -> Monet
    # G_Photo: Monet -> Photo
    G_Monet = ResNetGenerator().to(device)
    G_Photo = ResNetGenerator().to(device)
    
    D_Monet = PatchDiscriminator().to(device)
    D_Photo = PatchDiscriminator().to(device)

    # Initialize weights
    G_Monet.apply(weights_init_normal)
    G_Photo.apply(weights_init_normal)
    D_Monet.apply(weights_init_normal)
    D_Photo.apply(weights_init_normal)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_Monet.parameters(), G_Photo.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D_Monet = torch.optim.Adam(D_Monet.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_Photo = torch.optim.Adam(D_Photo.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - args.decay_epoch) / float(args.n_epochs - args.decay_epoch + 1)
    )
    lr_scheduler_D_Monet = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_Monet, lr_lambda=lambda epoch: 1.0 - max(0, epoch - args.decay_epoch) / float(args.n_epochs - args.decay_epoch + 1)
    )
    lr_scheduler_D_Photo = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_Photo, lr_lambda=lambda epoch: 1.0 - max(0, epoch - args.decay_epoch) / float(args.n_epochs - args.decay_epoch + 1)
    )

    # Buffers to store generated images
    fake_monet_buffer = ReplayBuffer()
    fake_photo_buffer = ReplayBuffer()

    # Dataloader
    transforms_ = get_transforms()
    dataset = ImageDataset(args.monet_path, args.photo_path, transforms_)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"Starting training for {args.n_epochs} epochs...")

    for epoch in range(args.start_epoch, args.n_epochs):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.n_epochs}")):
            
            # Set model input
            real_monet = batch['monet'].to(device)
            real_photo = batch['photo'].to(device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss
            # G_Monet(Monet) should be Monet
            loss_id_A = criterion_identity(G_Monet(real_monet), real_monet)
            # G_Photo(Photo) should be Photo
            loss_id_B = criterion_identity(G_Photo(real_photo), real_photo)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_monet = G_Monet(real_photo)
            loss_GAN_AB = criterion_GAN(D_Monet(fake_monet), torch.ones_like(D_Monet(fake_monet)))

            fake_photo = G_Photo(real_monet)
            loss_GAN_BA = criterion_GAN(D_Photo(fake_photo), torch.ones_like(D_Photo(fake_photo)))

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            rec_photo = G_Photo(fake_monet)
            loss_cycle_A = criterion_cycle(rec_photo, real_photo)

            rec_monet = G_Monet(fake_photo)
            loss_cycle_B = criterion_cycle(rec_monet, real_monet)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + (10.0 * loss_cycle) + (5.0 * loss_identity)
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_Monet.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_Monet(real_monet), torch.ones_like(D_Monet(real_monet)))
            # Fake loss (with buffer)
            fake_monet_ = fake_monet_buffer.push_and_pop(fake_monet)
            loss_fake = criterion_GAN(D_Monet(fake_monet_.detach()), torch.zeros_like(D_Monet(fake_monet_)))
            
            loss_D_Monet = (loss_real + loss_fake) / 2
            loss_D_Monet.backward()
            optimizer_D_Monet.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_Photo.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_Photo(real_photo), torch.ones_like(D_Photo(real_photo)))
            # Fake loss (with buffer)
            fake_photo_ = fake_photo_buffer.push_and_pop(fake_photo)
            loss_fake = criterion_GAN(D_Photo(fake_photo_.detach()), torch.zeros_like(D_Photo(fake_photo_)))

            loss_D_Photo = (loss_real + loss_fake) / 2
            loss_D_Photo.backward()
            optimizer_D_Photo.step()

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_Monet.step()
        lr_scheduler_D_Photo.step()

        # Save checkpoints every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"Saving checkpoint at epoch {epoch+1} to {args.checkpoint_dir}")
            torch.save(G_Monet.state_dict(), os.path.join(args.checkpoint_dir, f'G_Monet_epoch_{epoch+1}.pth'))
            torch.save(G_Photo.state_dict(), os.path.join(args.checkpoint_dir, f'G_Photo_epoch_{epoch+1}.pth'))
            torch.save(D_Monet.state_dict(), os.path.join(args.checkpoint_dir, f'D_Monet_epoch_{epoch+1}.pth'))
            torch.save(D_Photo.state_dict(), os.path.join(args.checkpoint_dir, f'D_Photo_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--decay_epoch', type=int, default=25, help='epoch from which to start lr decay')
    parser.add_argument('--monet_path', type=str, default='data/monet_jpg', help='path to monet images')
    parser.add_argument('--photo_path', type=str, default='data/photo_jpg', help='path to photo images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='path to save checkpoints')
    
    args = parser.parse_args()
    train(args)

