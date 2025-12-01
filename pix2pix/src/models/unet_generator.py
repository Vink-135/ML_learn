"""
Professional U-Net Generator for Pix2Pix
Implements advanced U-Net architecture with skip connections, attention mechanisms,
and feature preservation for photorealistic building facade generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionBlock(nn.Module):
    """Self-attention mechanism for better feature correlation"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention mechanism
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # Skip connection adjustment
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)

class DownBlock(nn.Module):
    """Encoder block with residual connections and attention"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(DownBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.residual = ResidualBlock(out_channels, out_channels)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        if self.attention:
            x = self.attention(x)
        return x

class UpBlock(nn.Module):
    """Decoder block with skip connections and feature fusion"""
    def __init__(self, in_channels, out_channels, skip_channels, use_attention=False, dropout=0.5):
        super(UpBlock, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.bn_up = nn.BatchNorm2d(out_channels)
        
        # Feature fusion for skip connection
        self.fusion = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.bn_fusion = nn.BatchNorm2d(out_channels)
        
        self.residual = ResidualBlock(out_channels, out_channels, dropout)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        
    def forward(self, x, skip):
        # Upsample
        x = F.relu(self.bn_up(self.upconv(x)))
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Feature fusion
        x = F.relu(self.bn_fusion(self.fusion(x)))
        
        # Residual processing
        x = self.residual(x)
        
        # Optional attention
        if self.attention:
            x = self.attention(x)
            
        return x

class UNetGenerator(nn.Module):
    """
    Professional U-Net Generator for Pix2Pix
    
    Features:
    - Skip connections for detail preservation
    - Attention mechanisms for better feature correlation
    - Residual blocks for stable training
    - Multi-scale feature processing
    - Dropout for regularization
    """
    
    def __init__(self, input_channels=3, output_channels=3, base_filters=64):
        super(UNetGenerator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder (Downsampling path)
        self.down1 = DownBlock(base_filters, base_filters * 2)           # 256 -> 128
        self.down2 = DownBlock(base_filters * 2, base_filters * 4)       # 128 -> 64
        self.down3 = DownBlock(base_filters * 4, base_filters * 8)       # 64 -> 32
        self.down4 = DownBlock(base_filters * 8, base_filters * 8)       # 32 -> 16
        self.down5 = DownBlock(base_filters * 8, base_filters * 8)       # 16 -> 8
        self.down6 = DownBlock(base_filters * 8, base_filters * 8)       # 8 -> 4
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 8, 4, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            AttentionBlock(base_filters * 8),
            ResidualBlock(base_filters * 8, base_filters * 8, dropout=0.5)
        )
        
        # Decoder (Upsampling path)
        self.up1 = UpBlock(base_filters * 8, base_filters * 8, base_filters * 8, use_attention=True, dropout=0.5)
        self.up2 = UpBlock(base_filters * 8, base_filters * 8, base_filters * 8, use_attention=True, dropout=0.5)
        self.up3 = UpBlock(base_filters * 8, base_filters * 8, base_filters * 8, use_attention=True, dropout=0.5)
        self.up4 = UpBlock(base_filters * 8, base_filters * 4, base_filters * 8, dropout=0.0)
        self.up5 = UpBlock(base_filters * 4, base_filters * 2, base_filters * 4, dropout=0.0)
        self.up6 = UpBlock(base_filters * 2, base_filters, base_filters * 2, dropout=0.0)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_filters, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Encoder path with skip connections
        e1 = self.initial(x)        # [B, 64, 256, 256]
        e2 = self.down1(e1)         # [B, 128, 128, 128]
        e3 = self.down2(e2)         # [B, 256, 64, 64]
        e4 = self.down3(e3)         # [B, 512, 32, 32]
        e5 = self.down4(e4)         # [B, 512, 16, 16]
        e6 = self.down5(e5)         # [B, 512, 8, 8]
        e7 = self.down6(e6)         # [B, 512, 4, 4]
        
        # Bottleneck
        bottleneck = self.bottleneck(e7)  # [B, 512, 2, 2]
        
        # Decoder path with skip connections
        d1 = self.up1(bottleneck, e7)     # [B, 512, 4, 4]
        d2 = self.up2(d1, e6)             # [B, 512, 8, 8]
        d3 = self.up3(d2, e5)             # [B, 512, 16, 16]
        d4 = self.up4(d3, e4)             # [B, 256, 32, 32]
        d5 = self.up5(d4, e3)             # [B, 128, 64, 64]
        d6 = self.up6(d5, e2)             # [B, 64, 128, 128]
        
        # Final output
        output = self.final(d6)           # [B, 3, 256, 256]
        
        return output

class HighResUNetGenerator(UNetGenerator):
    """High-resolution U-Net for 512x512 output"""
    
    def __init__(self, input_channels=3, output_channels=3, base_filters=64):
        super().__init__(input_channels, output_channels, base_filters)
        
        # Add extra layers for higher resolution
        self.down7 = DownBlock(base_filters * 8, base_filters * 8)
        self.up0 = UpBlock(base_filters * 8, base_filters * 8, base_filters * 8, use_attention=True, dropout=0.5)
        
    def forward(self, x):
        # Extended encoder for 512x512
        e1 = self.initial(x)        # [B, 64, 256, 256]
        e2 = self.down1(e1)         # [B, 128, 128, 128]
        e3 = self.down2(e2)         # [B, 256, 64, 64]
        e4 = self.down3(e3)         # [B, 512, 32, 32]
        e5 = self.down4(e4)         # [B, 512, 16, 16]
        e6 = self.down5(e5)         # [B, 512, 8, 8]
        e7 = self.down6(e6)         # [B, 512, 4, 4]
        e8 = self.down7(e7)         # [B, 512, 2, 2]
        
        # Bottleneck
        bottleneck = self.bottleneck(e8)  # [B, 512, 1, 1]
        
        # Extended decoder for 512x512
        d0 = self.up0(bottleneck, e8)     # [B, 512, 2, 2]
        d1 = self.up1(d0, e7)             # [B, 512, 4, 4]
        d2 = self.up2(d1, e6)             # [B, 512, 8, 8]
        d3 = self.up3(d2, e5)             # [B, 512, 16, 16]
        d4 = self.up4(d3, e4)             # [B, 256, 32, 32]
        d5 = self.up5(d4, e3)             # [B, 128, 64, 64]
        d6 = self.up6(d5, e2)             # [B, 64, 128, 128]
        
        # Final output
        output = self.final(d6)           # [B, 3, 256, 256]
        
        return output
