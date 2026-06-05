import torch
from torch import nn


class DomainCAENet(nn.Module):
    """
    Domain-Conditioned Autoencoder (Domain-CAE).

    Layout (matches the thesis spec):
        - Input: flat 640-D log-mel feature vector (n_mels * frames = 128 * 5).
        - CNN encoder operates on the reshaped (B, 1, n_mels=128, frames=5) tensor.
        - Dense bottleneck produces a small latent vector z.
        - A lightweight domain classifier head predicts P(target | x) from z.
        - The dense decoder is CONDITIONED on the domain signal:
            * train: true binary source/target label (when provided)
            * inference: predicted target probability
        - Reconstruction is a 640-D vector, identical to the baseline AE I/O.

    Mahalanobis buffers `cov_source` and `cov_target` are kept here so that
    `calc_inv_cov(model, ...)` from networks.criterion.mahala continues to work
    unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        block_size: int,
        n_mels: int = 128,
        frames: int = 5,
        latent_dim: int = 16,
    ):
        super().__init__()
        assert n_mels * frames == input_dim, (
            f"input_dim ({input_dim}) must equal n_mels*frames ({n_mels}*{frames})."
        )
        self.input_dim = input_dim
        self.n_mels = n_mels
        self.frames = frames
        self.latent_dim = latent_dim

        # Mahalanobis covariance buffers (kept as Parameters with requires_grad=False
        # to mirror the baseline AENet API used by calc_inv_cov).
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        # ---------------- CNN encoder ----------------
        # Input is reshaped to (B, 1, 128, 5).
        # We only downsample the mel axis (kernel/stride pool on dim=mels),
        # keeping the small frames axis intact.
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # -> (B,16,64,5)

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # -> (B,32,32,5)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 1)),      # -> (B,64,4,1)
        )
        self._conv_out_dim = 64 * 4 * 1  # 256

        # ---------------- Dense bottleneck ----------------
        self.fc_encoder = nn.Sequential(
            nn.Linear(self._conv_out_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

        # ---------------- Domain classifier head ----------------
        # Lightweight: latent -> hidden -> 1 logit. Sigmoid is applied at use sites.
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # ---------------- Conditional dense decoder ----------------
        # The decoder receives [z | domain_signal] where domain_signal is a single
        # scalar in [0,1] (true label during training, predicted prob at inference).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.n_mels, self.frames)
        h = self.conv_encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encoder(h)
        return z

    def classify_domain(self, z: torch.Tensor) -> torch.Tensor:
        return self.domain_head(z)  # logits, shape (B, 1)

    def decode(self, z: torch.Tensor, domain_signal: torch.Tensor) -> torch.Tensor:
        # domain_signal: (B, 1) in [0, 1]
        z_cond = torch.cat([z, domain_signal], dim=1)
        return self.decoder(z_cond)

    def forward(self, x: torch.Tensor, domain_label: torch.Tensor = None):
        """
        Args:
            x: input feature tensor, shape (B, 640) or (B, 1, 128, 5).
            domain_label: optional binary domain labels of shape (B,) or (B,1),
                1.0 = target, 0.0 = source. When provided AND the module is in
                training mode, the decoder is conditioned on this true label.
                Otherwise the decoder is conditioned on the predicted P(target).

        Returns:
            recon: (B, 640) reconstruction.
            z: (B, latent_dim) latent vector.
            domain_logits: (B, 1) raw logits from the domain classifier.
        """
        z = self.encode(x)
        domain_logits = self.classify_domain(z)
        domain_prob = torch.sigmoid(domain_logits)

        if (domain_label is not None) and self.training:
            cond = domain_label.float().view(-1, 1)
        else:
            cond = domain_prob

        recon = self.decode(z, cond)
        return recon, z, domain_logits
