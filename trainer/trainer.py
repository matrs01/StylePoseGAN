from typing import Dict, Tuple, Any
from pathlib import Path
import math

import torch
import torchvision
from torch.utils.data import DataLoader
import wandb
import numpy as np
from tqdm.auto import tqdm

from models.stylegan2 import Discriminator, Generator, GradientPenalty, \
    PathLengthPenalty
from models.ANet import ANet
from models.PNet import PNet
from models.losses import ReconstructionLoss, GeneratorLoss, DiscriminatorLoss
from helpers.helpers import cycle_dataloader
from helpers.dataset import FashionDataset


class Trainer:
    # [Gradient Penalty Regularization Loss](index.html#gradient_penalty)
    gradient_penalty = GradientPenalty()
    # Gradient penalty coefficient $\gamma$
    gradient_penalty_coefficient: float = 10.

    # [Path length penalty](index.html#path_length_penalty)
    path_length_penalty: PathLengthPenalty

    # Dimensionality of $z$ and $w$
    d_latent: int = 512
    # Generator & Discriminator learning rate
    learning_rate: float = 0.002
    pnet_learning_rate: float = 0.002
    anet_learning_rate: float = 0.002
    # Number of steps to accumulate gradients on. Use this to increase the effective batch size.
    gradient_accumulate_steps: int = 1
    # $\beta_1$ and $\beta_2$ for Adam optimizer
    adam_betas: Tuple[float, float] = (0.0, 0.99)
    # Probability of mixing styles
    style_mixing_prob: float = 0.9

    # Total number of training steps
    training_steps: int = 150_000

    # Number of blocks in the generator (calculated based on image resolution)
    n_gen_blocks: int

    # The interval at which to compute gradient penalty
    lazy_gradient_penalty_interval: int = 4
    # Path length penalty calculation interval
    lazy_path_penalty_interval: int = 32
    # Skip calculating path length penalty during the initial phase of training
    lazy_path_penalty_after: int = 5_000

    # How often to log generated images
    log_generated_interval: int = 500
    # How often to save model checkpoints
    save_checkpoint_interval: int = 2_000

    # Whether to log model layer outputs
    log_layer_outputs: bool = False

    sphere_face_net_dict: str = "pretrained/sphere20a.pth"

    def __init__(self, config: Dict):
        """
        ### Initialize
        """
        self.config = config
        self.batch_size = self.config["batch_size"]
        self.device = torch.device(self.config["device"])

        dataset = FashionDataset(self.config["data_path"])
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=32,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        self.loader = cycle_dataloader(dataloader)

        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.config["image_size"]))

        # Create discriminator and generator
        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(
            self.device)
        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create PNet
        self.pnet = PNet().to(self.device)
        # Create ANet
        self.anet = ANet(2 ** log_resolution, self.d_latent).to(self.device)
        # Create path length penalty loss
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        # Discriminator and generator losses
        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)
        self.reconstruction_loss = ReconstructionLoss().to(self.device)
        self.reconstruction_loss.face_loss.load_net(self.sphere_face_net_dict)
        self.reconstruction_loss.eval()

        # Create optimizers
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        gen_params = list(self.generator.parameters()) + \
                     list(self.pnet.parameters()) + \
                     list(self.anet.parameters())
        self.generator_optimizer = torch.optim.Adam(
            gen_params,
            lr=self.learning_rate, betas=self.adam_betas
        )

        self.global_step = 0
        self.pbar = tqdm(total=self.config["training_steps"])
        self.losses = dict()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])

        wandb.watch(self.generator, log_freq=100)

    def save_checkpoint(self,
                        save: bool = True
                        ) -> Dict[Any, Any]:
        checkpoint = {
            "gen_state_dict": self.generator.state_dict(),
            "disc_state_dict": self.discriminator.state_dict(),
            "pnet_state_dict": self.pnet.state_dict(),
            "anet_state_dict": self.anet.state_dict(),
            "gen_opt_state_dict": self.generator_optimizer.state_dict(),
            "disc_opt_state_dict": self.discriminator_optimizer.state_dict(),
            "global_step": self.global_step,
        }

        path = Path() / self.config["save_checkpoint_path"] \
               / f"step={self.global_step}.pt"

        if save:
            torch.save(checkpoint, path)
        return checkpoint

    def load_checkpoint(self,
                        checkpoint_path: Path,
                        ) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint[
                                               "discriminator_state_dict"])
        self.pnet.load_state_dict(checkpoint["pnet_state_dict"])
        self.anet.load_state_dict(checkpoint["anet_state_dict"])

        self.generator_optimizer.load_state_dict(checkpoint[
                                                     "gen_opt_state_dict"])
        self.discriminator_optimizer.load_state_dict(checkpoint[
                                                         "disc_opt_state_dict"])

        self.global_step = checkpoint["global_step"] + 1
        self.pbar.update(self.global_step)

    def get_noise(self, batch_size: int):
        """
        ### Generate noise

        This generates noise for each [generator block](index.html#generator_block)
        """
        # List to store noise
        noise = []
        # Noise resolution starts from $4$
        resolution = 4

        # Generate noise for each generator block
        for i in range(self.n_gen_blocks):
            # The first block has only one $3 \times 3$ convolution
            if i == 0:
                n1 = None
            # Generate noise to add after the first convolution layer
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution,
                                 device=self.device)
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution,
                             device=self.device)

            # Add noise tensors to the list
            noise.append((n1, n2))

            # Next block has $2 \times$ resolution
            resolution *= 2

        # Return noise tensors
        return noise

    def generate_images(self, s_pose, s_app, t_pose):
        """
        ### Generate images

        This generate images using the generator
        """

        batch_size = s_pose.shape[0]

        s_app_enc = self.anet(s_app)
        s_pose_enc = self.pnet(s_pose)
        t_pose_enc = self.pnet(t_pose)
        # Get noise
        noise = self.get_noise(batch_size)

        s_app_enc = s_app_enc[None, :, :].expand(self.n_gen_blocks, -1, -1)

        # Generate images
        restored = self.generator(s_pose_enc, s_app_enc, noise)
        # TODO: same noise?
        transferred = self.generator(t_pose_enc, s_app_enc, noise)

        # Return images and $w$
        return restored, transferred, s_app_enc

    def discriminator_step(self):
        self.discriminator_optimizer.zero_grad()

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.gradient_accumulate_steps):
            s_real, s_pose, s_app, t_real, t_pose = next(self.loader)
            s_real = self.transforms(s_real).to(self.device)
            s_pose = self.transforms(s_pose).to(self.device)
            s_app = self.transforms(s_app).to(self.device)
            t_real = self.transforms(t_real).to(self.device)
            t_pose = self.transforms(t_pose).to(self.device)

            # Sample images from generator
            s_restored, s_transferred, s_app_enc = \
                self.generate_images(s_pose, s_app, t_pose)
            # Discriminator classification for generated images
            s_fake_output = self.discriminator(s_restored)
            t_fake_output = self.discriminator(s_transferred)

            # We need to calculate gradients w.r.t. real images for gradient penalty
            if self.global_step % self.lazy_gradient_penalty_interval == 1:
                s_real.requires_grad_()
                t_real.requires_grad_()
            # Discriminator classification for real images
            s_real_output = self.discriminator(s_real)
            t_real_output = self.discriminator(t_real)

            # Get discriminator loss
            s_real_loss, s_fake_loss = self.discriminator_loss(s_real_output,
                                                               s_fake_output)
            t_real_loss, t_fake_loss = self.discriminator_loss(t_real_output,
                                                               t_fake_output)
            disc_loss = s_real_loss + s_fake_loss + t_real_loss + t_fake_loss

            # Add gradient penalty
            if self.global_step % self.lazy_gradient_penalty_interval == 1:
                # Calculate and log gradient penalty
                gp = self.gradient_penalty(s_real, s_real_output) + \
                     self.gradient_penalty(t_real, t_real_output)
                # Multiply by coefficient and add gradient penalty
                disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

            # Compute gradients
            disc_loss.backward()

            self.losses["disc_loss"] = disc_loss.item()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                       max_norm=1.0)
        # Take optimizer step
        self.discriminator_optimizer.step()

    def generator_step(self):
        self.generator_optimizer.zero_grad()

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.gradient_accumulate_steps):
            s_real, s_pose, s_app, t_real, t_pose = next(self.loader)
            s_real = self.transforms(s_real).to(self.device)
            s_pose = self.transforms(s_pose).to(self.device)
            s_app = self.transforms(s_app).to(self.device)
            t_real = self.transforms(t_real).to(self.device)
            t_pose = self.transforms(t_pose).to(self.device)

            # Sample images from generator
            s_restored, s_transferred, s_app_enc = \
                self.generate_images(s_pose, s_app, t_pose)
            # Discriminator classification for generated images
            fake_output = self.discriminator(s_restored) + \
                          self.discriminator(s_transferred)

            # Get generator loss
            gen_loss = self.generator_loss(fake_output) + \
                       self.reconstruction_loss(s_real, s_restored) + \
                       self.reconstruction_loss(t_real, s_transferred)

            # Calculate gradients
            gen_loss.backward()

            self.losses["gen_loss"] = gen_loss.item()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                       max_norm=1.0)

        self.generator_optimizer.step()

        if (self.global_step + 10) % self.config["pictures_log_period"] == 0:
            self._show_pictures(s_real[0], t_real[0], s_restored[0],
                                s_transferred[0])

    def _show_pictures(self, a, b, c, d):
        a = a.detach().cpu()
        b = b.detach().cpu()
        c = c.detach().cpu()
        d = d.detach().cpu()
        images = (torchvision.utils.make_grid([a, b, c, d],
                                              nrow=4,
                                              normalize=False)
                  .permute(1, 2, 0) * torch.Tensor([0.229, 0.224, 0.225])
                  + torch.Tensor([0.485, 0.456, 0.406])).numpy()

        np.nan_to_num(images, copy=False, nan=1)
        images = images.clip(0, 1)

        wandb.log({"generated images": [wandb.Image(images)]})

    def update_logs_(self):
        self.pbar.set_postfix(self.losses)
        if self.global_step % self.config["wandb_log_period"] == 1:
            wandb.log(self.losses)

    def step(self):
        self.discriminator_step()
        self.generator_step()

        # Save model checkpoints
        if self.global_step % self.config["save_checkpoint_period"] == 10:
            self.save_checkpoint()

    def train(self):
        for i in range(self.config["training_steps"]):
            self.step()
            self.update_logs_()
            self.global_step += 1
            self.pbar.update(1)
        self.pbar.close()
