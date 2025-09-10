from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch.nn.functional as func

class causalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    
    def forward(self, x):
        x = func.pad(x, (self.kernel_size - 1, 0))                          # pad left only
        return self.conv(x)

class TCN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)

        in_c = observation_space.shape[1]                                   # number of features in obs
        timesteps = observation_space.shape[0]                              # number of steps in obs for normalization

        # TCN architecture
        self.tcn = nn.Sequential(
            
            # projection layer first (in_c x 1 conv)
            nn.Conv1d(in_channels=in_c, out_channels=features_dim, kernel_size=1),

            # 7 1D convolutional layers with layer normalization and tanh as activation function
            # RF = 31 (L = 7, K = 5)
            causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh(),
            
            causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh(),
            
            causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh(),

            causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh(),

            causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh(),

            causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh(),

            causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh()
        )

        # project (last timestep) into a latent vector for the actor and critic MLPs
        self.head = nn.Sequential(
            nn.Linear(in_features=features_dim, out_features=features_dim)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)          # [B, T, C] -> [B, C, T]
        x = self.tcn(x)                 
        x = x[:, :, -1]                 # last timestep [B, channels]
        return self.head(x)               
    
