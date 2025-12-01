"""
Professional PatchGAN Discriminator for Pix2Pix
Implements multi-scale PatchGAN architecture for realistic texture generation
and high-quality adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralNorm(nn.Module):
    """Spectral normalization for stable training"""
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class DiscriminatorBlock(nn.Module):
    """Discriminator block with spectral normalization and leaky ReLU"""
    def __init__(self, in_channels, out_channels, stride=2, use_spectral_norm=True, use_batch_norm=True):
        super(DiscriminatorBlock, self).__init__()
        
        conv = nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)
        
        if use_spectral_norm:
            conv = SpectralNorm(conv)
        
        layers = [conv]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix
    
    Features:
    - Multi-scale patch-based discrimination
    - Spectral normalization for stable training
    - Receptive field covers 70x70 patches
    - Conditional input (semantic + realistic images)
    """
    
    def __init__(self, input_channels=6, base_filters=64, n_layers=3, use_spectral_norm=True):
        super(PatchGANDiscriminator, self).__init__()
        
        # Initial layer (no batch norm)
        self.initial = DiscriminatorBlock(
            input_channels, base_filters, 
            stride=2, use_spectral_norm=use_spectral_norm, use_batch_norm=False
        )
        
        # Progressive layers
        layers = []
        in_channels = base_filters
        
        for i in range(n_layers):
            out_channels = min(base_filters * (2 ** (i + 1)), 512)
            stride = 2 if i < n_layers - 1 else 1
            
            layers.append(DiscriminatorBlock(
                in_channels, out_channels,
                stride=stride, use_spectral_norm=use_spectral_norm
            ))
            
            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        
        # Final classification layer
        final_conv = nn.Conv2d(in_channels, 1, 4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = SpectralNorm(final_conv)
        
        self.final = final_conv
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, semantic_img, realistic_img):
        # Concatenate semantic and realistic images
        x = torch.cat([semantic_img, realistic_img], dim=1)
        
        # Pass through discriminator layers
        x = self.initial(x)
        x = self.layers(x)
        x = self.final(x)
        
        return x

class MultiScalePatchGANDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN Discriminator
    
    Uses multiple discriminators at different scales for better
    texture quality and more stable training.
    """
    
    def __init__(self, input_channels=6, base_filters=64, num_scales=3):
        super(MultiScalePatchGANDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        # Create discriminators for different scales
        for i in range(num_scales):
            # Reduce number of layers for smaller scales
            n_layers = max(2, 4 - i)
            discriminator = PatchGANDiscriminator(
                input_channels=input_channels,
                base_filters=base_filters,
                n_layers=n_layers
            )
            self.discriminators.append(discriminator)
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, semantic_img, realistic_img):
        results = []
        
        # Current scale inputs
        semantic_current = semantic_img
        realistic_current = realistic_img
        
        for i, discriminator in enumerate(self.discriminators):
            # Apply discriminator at current scale
            result = discriminator(semantic_current, realistic_current)
            results.append(result)
            
            # Downsample for next scale (except for last)
            if i < self.num_scales - 1:
                semantic_current = self.downsample(semantic_current)
                realistic_current = self.downsample(realistic_current)
        
        return results

class FeatureMatchingDiscriminator(PatchGANDiscriminator):
    """
    Enhanced discriminator that returns intermediate features
    for feature matching loss computation.
    """
    
    def __init__(self, input_channels=6, base_filters=64, n_layers=3, use_spectral_norm=True):
        super().__init__(input_channels, base_filters, n_layers, use_spectral_norm)
    
    def forward(self, semantic_img, realistic_img, return_features=False):
        # Concatenate inputs
        x = torch.cat([semantic_img, realistic_img], dim=1)
        
        features = []
        
        # Initial layer
        x = self.initial(x)
        if return_features:
            features.append(x)
        
        # Progressive layers
        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)
        
        # Final layer
        output = self.final(x)
        
        if return_features:
            return output, features
        else:
            return output

class ProgressiveDiscriminator(nn.Module):
    """
    Progressive discriminator that can handle multiple resolutions
    for progressive training from low to high resolution.
    """
    
    def __init__(self, input_channels=6, base_filters=64, max_layers=6):
        super(ProgressiveDiscriminator, self).__init__()
        
        self.max_layers = max_layers
        self.current_layers = 1
        
        # Build all possible layers
        self.layers = nn.ModuleList()
        
        # Initial layer
        self.layers.append(DiscriminatorBlock(
            input_channels, base_filters,
            stride=2, use_batch_norm=False
        ))
        
        # Progressive layers
        in_channels = base_filters
        for i in range(1, max_layers):
            out_channels = min(base_filters * (2 ** i), 512)
            self.layers.append(DiscriminatorBlock(
                in_channels, out_channels,
                stride=2 if i < max_layers - 1 else 1
            ))
            in_channels = out_channels
        
        # Final layers for each resolution
        self.final_layers = nn.ModuleList()
        for i in range(max_layers):
            channels = min(base_filters * (2 ** i), 512)
            self.final_layers.append(nn.Conv2d(channels, 1, 4, stride=1, padding=1))
    
    def grow_network(self):
        """Add one more layer to the discriminator"""
        if self.current_layers < self.max_layers:
            self.current_layers += 1
    
    def forward(self, semantic_img, realistic_img):
        x = torch.cat([semantic_img, realistic_img], dim=1)
        
        # Pass through active layers
        for i in range(self.current_layers):
            x = self.layers[i](x)
        
        # Apply appropriate final layer
        output = self.final_layers[self.current_layers - 1](x)
        
        return output

def create_discriminator(discriminator_type="patchgan", **kwargs):
    """Factory function to create different types of discriminators"""
    
    if discriminator_type == "patchgan":
        return PatchGANDiscriminator(**kwargs)
    elif discriminator_type == "multiscale":
        return MultiScalePatchGANDiscriminator(**kwargs)
    elif discriminator_type == "feature_matching":
        return FeatureMatchingDiscriminator(**kwargs)
    elif discriminator_type == "progressive":
        return ProgressiveDiscriminator(**kwargs)
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")

# Test function
if __name__ == "__main__":
    # Test discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test inputs
    semantic = torch.randn(2, 3, 256, 256).to(device)
    realistic = torch.randn(2, 3, 256, 256).to(device)
    
    # Test PatchGAN
    discriminator = PatchGANDiscriminator().to(device)
    output = discriminator(semantic, realistic)
    print(f"PatchGAN output shape: {output.shape}")
    
    # Test Multi-scale
    multi_discriminator = MultiScalePatchGANDiscriminator().to(device)
    outputs = multi_discriminator(semantic, realistic)
    print(f"Multi-scale outputs: {[out.shape for out in outputs]}")
    
    print("Discriminator tests passed!")
