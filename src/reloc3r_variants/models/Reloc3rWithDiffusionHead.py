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


def compute_gt_poses(view1, view2):
    """
    Compute the two-way relative camera poses between two views.
    Each pose is a vector of length 12, where the first 9 elements are flattened rotation matrix and the last 3 elements are translation vector.
    """
    abs_pose1, abs_pose2 = view1["camera_pose"], view2["camera_pose"]
    gt_pose1to2 = torch.inverse(view2['camera_pose']) @ view1['camera_pose'] # (B, 4, 4)
    gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose'] # (B, 4, 4)

    gt_pose1to2_reshape = torch.zeros(abs_pose1.shape[0], 12, device=abs_pose1.device)
    gt_pose1to2_reshape[:, :9] = gt_pose1to2[:, :3, :3].reshape(abs_pose1.shape[0], 9)
    gt_pose1to2_reshape[:, 9:] = gt_pose1to2[:, :3, 3]

    gt_pose2to1_reshape = torch.zeros(abs_pose2.shape[0], 12, device=abs_pose2.device)
    gt_pose2to1_reshape[:, :9] = gt_pose2to1[:, :3, :3].reshape(abs_pose2.shape[0], 9)
    gt_pose2to1_reshape[:, 9:] = gt_pose2to1[:, :3, 3]

    return gt_pose1to2_reshape, gt_pose2to1_reshape


class Reloc3rWithDiffusionHead(nn.Module):
    def __init__(self, img_size=512):
        super().__init__()

        self.backbone = Reloc3rRelpose_Almost(img_size=img_size)
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
        """
        Compute the diffusion loss for training
        """
        # Embed image features
        out1, out2 = self.backbone(view1, view2)
        image_features1, image_features2 = out1["features"], out2["features"]

        # Compute gt_poses
        gt_pose1to2, gt_pose2to1 = compute_gt_poses(view1, view2)

        # Compute loss
        loss1 = self.compute_loss(image_features1, gt_pose1to2)
        loss2 = self.compute_loss(image_features2, gt_pose2to1)

        return loss1 + loss2
    

    def sample_pose(self, view1, view2, num_steps: int = 50):
        """
        Sample a pose from using the learned velocity model.
        Here we only sample the pose2to1
        """
        # Embed image features ahead
        out1, out2 = self.backbone(view1, view2)
        image_features1, image_features2 = out1["features"], out2["features"]

        # Define diffusion components
        dt = -1.0 / num_steps
        batch_size = image_features1.shape[0]
        noise = torch.randn(batch_size, self.pose_dim, device=image_features2.device)

        # Step function
        def step(x_t, t):
            """
            :param x_t: a torch.Tensor of shape (B, 12)
            :param t: a torch.Tensor of shape (B,). All elements should be the same (a multiple of dt)
            """
            # Embed timesteps
            timesteps_embed = posemb_sincos(t, embedding_dim=self.output_dim, min_period=4e-3, max_period=4.0)

            # Predict velocity
            v_t = self.velocity_prediction(
                image_embed=image_features2,
                noisy_pose=x_t,
                timesteps_embed=timesteps_embed
            )  # (B, pose_dim)

            return x_t + dt * v_t, t + dt
        
        # Begin sampling
        t = torch.ones(batch_size, device=image_features2.device)
        x_t = noise
        for _ in range(num_steps):
            x_t, t = step(x_t, t)

        return x_t
