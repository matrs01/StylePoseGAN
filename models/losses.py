import torch.utils.data
from torch.nn import functional as F

from labml_helpers.module import Module
from DISTS_pytorch import DISTS

from models.net_sphere import FaceNetWithUpsample


class FaceIdentityLoss(Module):
    def __init__(self):
        super().__init__()
        self.net = FaceNetWithUpsample()
        self.net.eval()

    def load_net(self, path):
        self.net.load_net(path)

    def __call__(self, real: torch.Tensor, restored: torch.Tensor):
        return F.l1_loss(self.net(real), self.net(restored))


class ReconstructionLoss(Module):
    def __init__(self):
        super().__init__()
        self.vgg_loss = DISTS()
        self.face_loss = FaceIdentityLoss()

    def __call__(self, real: torch.Tensor, restored: torch.Tensor):
        l1_loss = F.l1_loss(real, restored)
        vgg_loss = self.vgg_loss(real, restored, require_grad=True,
                                 batch_average=True)
        face_loss = self.face_loss(real, restored)
        return l1_loss + vgg_loss + face_loss


class DiscriminatorLoss(Module):
    """
    ## Discriminator Loss
    We want to find $w$ to maximize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big(x^{(i)} \big) +
     \frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def __call__(self, f_real: torch.Tensor, f_fake: torch.Tensor):
        """
        * `f_real` is $f_w(x)$
        * `f_fake` is $f_w(g_\theta(z))$
        This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
        which are later added.
        They are kept separate for logging.
        """

        # We use ReLUs to clip the loss to keep $f \in [-1, +1]$ range.
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


class GeneratorLoss(Module):
    """
    ## Generator Loss
    We want to find $\theta$ to minimize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$
    The first component is independent of $\theta$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def __call__(self, f_fake: torch.Tensor):
        """
        * `f_fake` is $f_w(g_\theta(z))$
        """
        return -f_fake.mean()
