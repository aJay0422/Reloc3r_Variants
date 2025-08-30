from .reloc3r_modules.reloc3r_encoder_decoder import Reloc3rRelpose_Almost

import torch
import torch.nn as nn


def posemb_sincos(
        timesteps, embedding_dim, min_period, max_period
):
    """
    Do sine-cosine positional embedding for a sequence of timesteps.
    :param timesteps: torch.Tensor of shape (batch_size,)
    :param embedding_dim: an integer
    :param min_period: a float
    :param max_period: a float
    """
    assert embedding_dim % 2 == 0, "Embedding dimension must be even"
    device = timesteps.device
    batch_size = timesteps.shape[0]
    

    # Create the position embeddings
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    period = period.to(device)

    pos_emb = torch.zeros(batch_size, embedding_dim, device=device)
    pos_emb[:, 0::2] = torch.sin(timesteps[:, None] * period[None, :])
    pos_emb[:, 1::2] = torch.cos(timesteps[:, None] * period[None, :])

    return pos_emb


class Reloc3rWithDiffusionHead(nn.Module):
    def __init__(self,):
        super().__init__()

        self.backbone = Reloc3rRelpose_Almost(img_size=512)
        self.output_dim = self.backbone.pose_head.proj.out_features
        self.pose_dim = 12

        self.pose_in_proj = nn.Linear(self.pose_dim, self.output_dim)
        self.pose_time_mlp = nn.Linear(2 * self.output_dim, self.output_dim)
        self.image_pose_time_mlp = nn.Linear(2 * self.output_dim, self.output_dim)
        self.pose_out_proj = nn.Linear(self.output_dim, self.pose_dim)

    def velocity_prediction(
        self,
        image_embed: torch.Tensor,
        noisy_pose: torch.Tensor,
        timesteps_embed: torch.Tensor,
    ):
        """
        Predict the velocity given image features, noisy pose, and timesteps embedding.
        :param image_embed: a torch.Tensor of shape (B, output_dim)
        :param noisy_pose: a torch.Tensor of shape (B, pose_dim)
        :param timesteps_embed: a torch.Tensor of shape (B, output_dim)
        :return: a torch.Tensor of shape (B, pose_dim)
        """
        # Embed noisy pose
        pose_embed = self.pose_in_proj(noisy_pose)  # (B, output_dim)

        # Embed pose and time
        pose_time_embed = self.pose_time_mlp(
            torch.cat([pose_embed, timesteps_embed], dim=-1)
        )
        pose_time_embed = torch.nn.functional.silu(pose_time_embed)

        # Embed image and pose/time
        image_pose_time_embed = self.image_pose_time_mlp(
            torch.cat([image_embed, pose_time_embed], dim=-1)
        )
        image_pose_time_embed = torch.nn.functional.silu(image_pose_time_embed)

        # Predict velocity
        v_t = self.pose_out_proj(image_pose_time_embed)  # (B, pose_dim)

        return v_t


    def compute_loss(
        self,
        image_features: torch.Tensor,
        gt_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param image_features: a torch.Tensor of shape (B, D)
        :param gt_pose: a torch.Tensor of shape (B, 12)
        """
        assert image_features.shape[0] == gt_pose.shape[0]
        batch_size = image_features.shape[0]

        # Add noise to gt_pose
        noise = torch.randn_like(gt_pose)
        timesteps = torch.distributions.Beta(1.5, 1.0).sample((batch_size,)) * 0.999 + 0.001 # Following pi0 implementation: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py#L249
        timesteps = timesteps.to(image_features.device)
        timesteps_expanded = timesteps.unsqueeze(-1)
        x_t = timesteps_expanded * noise + (1 - timesteps_expanded) * gt_pose
        u_t = noise - gt_pose # This is the ground truth velocity

        # Positional embedding for timesteps
        timesteps_embed = posemb_sincos(timesteps, embedding_dim=self.output_dim, min_period=4e-3, max_period=4.0)

        # Predict velocity
        v_t = self.velocity_prediction(
            image_embed=image_features,
            noisy_pose=x_t,
            timesteps_embed=timesteps_embed
        ) # (B, pose_dim)

        # Compute loss
        loss = torch.nn.functional.mse_loss(v_t, u_t, reduction="mean")

        return loss
    

    def forward(self, view1, view2):
        # Embed image features
        out1, out2 = self.backbone(view1, view2)
        image_features1, image_features2 = out1["features"], out2["features"]

        raise NotImplementedError("Loss computation not implemented")

