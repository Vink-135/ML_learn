"""
Professional Pix2Pix GAN Implementation for Photorealistic Building Facade Generation

This module implements a state-of-the-art Pix2Pix architecture with:
- Enhanced U-Net generator with attention mechanisms
- Multi-scale PatchGAN discriminator
- Advanced loss functions for photorealistic results
- Optimized for architectural facade generation
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal

# Configuration
OUTPUT_CHANNELS = 3
LAMBDA_L1 = 100  # Weight for L1 loss
LAMBDA_PERCEPTUAL = 10  # Weight for perceptual loss

class SpectralNormalization(layers.Layer):
    """Spectral normalization for stable GAN training"""
    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.layer.build(input_shape)

        # Get weight shape
        weight = self.layer.kernel
        weight_shape = weight.shape

        # Initialize u and v vectors
        self.u = self.add_weight(
            shape=(1, weight_shape[-1]),
            initializer='random_normal',
            trainable=False,
            name='u'
        )
        self.v = self.add_weight(
            shape=(1, np.prod(weight_shape[:-1])),
            initializer='random_normal',
            trainable=False,
            name='v'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Perform power iteration
        weight = self.layer.kernel
        weight_reshaped = tf.reshape(weight, [-1, weight.shape[-1]])

        u = self.u
        v = self.v

        for _ in range(self.power_iterations):
            v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(weight_reshaped)))
            u = tf.nn.l2_normalize(tf.matmul(v, weight_reshaped))

        # Compute spectral norm
        sigma = tf.matmul(tf.matmul(v, weight_reshaped), tf.transpose(u))

        # Normalize weights
        self.layer.kernel.assign(weight / sigma)

        # Update u and v
        self.u.assign(u)
        self.v.assign(v)

        return self.layer(inputs)

class AttentionBlock(layers.Layer):
    """Self-attention mechanism for better feature correlation"""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.query_conv = layers.Conv2D(channels // 8, 1)
        self.key_conv = layers.Conv2D(channels // 8, 1)
        self.value_conv = layers.Conv2D(channels, 1)
        self.gamma = self.add_weight(
            shape=(),
            initializer='zeros',
            trainable=True,
            name='gamma'
        )

    def call(self, x):
        batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Generate query, key, value
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Reshape for matrix multiplication
        query = tf.reshape(query, [batch_size, height * width, -1])
        key = tf.reshape(key, [batch_size, height * width, -1])
        value = tf.reshape(value, [batch_size, height * width, -1])

        # Attention mechanism
        attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True))
        out = tf.matmul(attention, value)
        out = tf.reshape(out, [batch_size, height, width, channels])

        # Apply learnable weight and residual connection
        return self.gamma * out + x

def enhanced_downsample(filters, size=4, apply_batchnorm=True, apply_spectral_norm=False):
    """Enhanced downsampling block with optional spectral normalization"""
    initializer = RandomNormal(0., 0.02)

    result = tf.keras.Sequential()

    conv = layers.Conv2D(filters, size, strides=2, padding='same',
                        kernel_initializer=initializer, use_bias=False)

    if apply_spectral_norm:
        conv = SpectralNormalization(conv)

    result.add(conv)

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU(0.2))

    return result

def enhanced_upsample(filters, size=4, apply_dropout=False, dropout_rate=0.5):
    """Enhanced upsampling block with configurable dropout"""
    initializer = RandomNormal(0., 0.02)

    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(dropout_rate))

    result.add(layers.ReLU())

    return result

class ResidualBlock(layers.Layer):
    """Residual block for better gradient flow"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.3)

    def build(self, input_shape):
        super().build(input_shape)
        # Create skip connection adjustment if needed
        input_channels = input_shape[-1]
        if input_channels != self.filters:
            self.skip_conv = layers.Conv2D(self.filters, 1, use_bias=False)
            self.skip_bn = layers.BatchNormalization()
        else:
            self.skip_conv = None
            self.skip_bn = None

    def call(self, x, training=None):
        residual = x

        # Adjust residual if channel dimensions don't match
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
            residual = self.skip_bn(residual, training=training)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Add residual connection
        x = x + residual
        return tf.nn.relu(x)

def ProfessionalGenerator(input_shape=(512, 512, 3), base_filters=64):
    """
    Professional U-Net Generator with enhanced features:
    - Skip connections for detail preservation
    - Attention mechanisms for better feature correlation
    - Stable architecture without dimension mismatches
    - Multi-scale processing for photorealistic results
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder (Downsampling path)
    # Layer 1: 512 -> 256
    e1 = enhanced_downsample(base_filters, apply_batchnorm=False)(inputs)

    # Layer 2: 256 -> 128
    e2 = enhanced_downsample(base_filters * 2)(e1)

    # Layer 3: 128 -> 64
    e3 = enhanced_downsample(base_filters * 4)(e2)

    # Layer 4: 64 -> 32
    e4 = enhanced_downsample(base_filters * 8)(e3)

    # Layer 5: 32 -> 16
    e5 = enhanced_downsample(base_filters * 8)(e4)

    # Layer 6: 16 -> 8
    e6 = enhanced_downsample(base_filters * 8)(e5)

    # Layer 7: 8 -> 4
    e7 = enhanced_downsample(base_filters * 8)(e6)

    # Layer 8: 4 -> 2 (Bottleneck)
    e8 = enhanced_downsample(base_filters * 8)(e7)

    # Bottleneck with attention
    bottleneck = AttentionBlock(base_filters * 8)(e8)

    # Decoder (Upsampling path with skip connections)
    # Layer 1: 2 -> 4
    d1 = enhanced_upsample(base_filters * 8, apply_dropout=True)(bottleneck)
    d1 = layers.Concatenate()([d1, e7])

    # Layer 2: 4 -> 8
    d2 = enhanced_upsample(base_filters * 8, apply_dropout=True)(d1)
    d2 = layers.Concatenate()([d2, e6])

    # Layer 3: 8 -> 16
    d3 = enhanced_upsample(base_filters * 8, apply_dropout=True)(d2)
    d3 = layers.Concatenate()([d3, e5])
    d3 = AttentionBlock(base_filters * 16)(d3)  # Attention on concatenated features

    # Layer 4: 16 -> 32
    d4 = enhanced_upsample(base_filters * 8)(d3)
    d4 = layers.Concatenate()([d4, e4])

    # Layer 5: 32 -> 64
    d5 = enhanced_upsample(base_filters * 4)(d4)
    d5 = layers.Concatenate()([d5, e3])

    # Layer 6: 64 -> 128
    d6 = enhanced_upsample(base_filters * 2)(d5)
    d6 = layers.Concatenate()([d6, e2])

    # Layer 7: 128 -> 256
    d7 = enhanced_upsample(base_filters)(d6)
    d7 = layers.Concatenate()([d7, e1])

    # Final layer: 256 -> 512
    initializer = RandomNormal(0., 0.02)
    final = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')(d7)

    return Model(inputs=inputs, outputs=final, name='ProfessionalGenerator')

# Backward compatibility
def Generator():
    """Backward compatible generator function"""
    return ProfessionalGenerator(input_shape=(256, 256, 3))

def ProfessionalPatchGANDiscriminator(input_shape=(512, 512, 3), base_filters=64):
    """
    Professional PatchGAN Discriminator with enhanced features:
    - Multi-scale patch-based discrimination
    - Spectral normalization for stable training
    - Receptive field covers 70x70 patches
    - Improved architecture for photorealistic results
    """
    initializer = RandomNormal(0., 0.02)

    # Input layers
    input_image = layers.Input(shape=input_shape, name='input_image')
    target_image = layers.Input(shape=input_shape, name='target_image')

    # Concatenate input and target
    x = layers.Concatenate()([input_image, target_image])  # (bs, H, W, 6)

    # Layer 1: No batch norm in first layer
    x = enhanced_downsample(base_filters, apply_batchnorm=False, apply_spectral_norm=True)(x)

    # Layer 2
    x = enhanced_downsample(base_filters * 2, apply_spectral_norm=True)(x)

    # Layer 3
    x = enhanced_downsample(base_filters * 4, apply_spectral_norm=True)(x)

    # Layer 4: Stride 1 for final layers
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(base_filters * 8, 4, strides=1,
                     kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Final classification layer
    x = layers.ZeroPadding2D()(x)
    output = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)

    return Model(inputs=[input_image, target_image], outputs=output,
                name='ProfessionalPatchGANDiscriminator')

class MultiScaleDiscriminator(Model):
    """Multi-scale discriminator for better texture quality"""
    def __init__(self, input_shape=(512, 512, 3), num_scales=3, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales
        self.discriminators = []

        # Create discriminators for different scales
        for i in range(num_scales):
            disc = ProfessionalPatchGANDiscriminator(input_shape)
            self.discriminators.append(disc)

        # Downsampling layer for multi-scale
        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

    def call(self, inputs):
        input_image, target_image = inputs
        results = []

        # Current scale inputs
        input_current = input_image
        target_current = target_image

        for i, discriminator in enumerate(self.discriminators):
            # Apply discriminator at current scale
            result = discriminator([input_current, target_current])
            results.append(result)

            # Downsample for next scale (except for last)
            if i < self.num_scales - 1:
                input_current = self.downsample(input_current)
                target_current = self.downsample(target_current)

        return results

# Backward compatibility
def Discriminator():
    """Backward compatible discriminator function"""
    return ProfessionalPatchGANDiscriminator(input_shape=(256, 256, 3))

# Advanced Loss Functions
class PerceptualLoss(layers.Layer):
    """Perceptual loss using VGG19 features for photorealistic results"""
    def __init__(self, use_pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pretrained = use_pretrained

        if use_pretrained:
            try:
                # Load pre-trained VGG19
                vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
                vgg.trainable = False

                # Extract features from multiple layers
                self.feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
                outputs = [vgg.get_layer(name).output for name in self.feature_layers]
                self.vgg_model = Model(vgg.input, outputs)
                self.has_vgg = True
            except Exception as e:
                print(f"Warning: Could not load VGG19: {e}")
                self.has_vgg = False
        else:
            self.has_vgg = False

    def call(self, y_true, y_pred):
        if self.has_vgg:
            # Preprocess images for VGG (scale to [0, 255] and apply VGG preprocessing)
            y_true_processed = tf.keras.applications.vgg19.preprocess_input((y_true + 1) * 127.5)
            y_pred_processed = tf.keras.applications.vgg19.preprocess_input((y_pred + 1) * 127.5)

            # Extract features
            true_features = self.vgg_model(y_true_processed)
            pred_features = self.vgg_model(y_pred_processed)

            # Compute perceptual loss
            loss = 0
            for true_feat, pred_feat in zip(true_features, pred_features):
                loss += tf.reduce_mean(tf.abs(true_feat - pred_feat))

            return loss
        else:
            # Fallback to simple feature loss
            return tf.reduce_mean(tf.abs(y_true - y_pred))

def generator_loss(disc_generated_output, gen_output, target, perceptual_loss_fn=None):
    """
    Enhanced generator loss combining adversarial, L1, and perceptual losses
    """
    # Adversarial loss
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output)

    # L1 loss for pixel-level accuracy
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Perceptual loss for photorealistic quality
    perceptual_loss = 0
    if perceptual_loss_fn is not None:
        perceptual_loss = perceptual_loss_fn(target, gen_output)

    # Total generator loss
    total_gen_loss = gan_loss + (LAMBDA_L1 * l1_loss) + (LAMBDA_PERCEPTUAL * perceptual_loss)

    return total_gen_loss, gan_loss, l1_loss, perceptual_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """Enhanced discriminator loss with label smoothing"""
    # Real loss with label smoothing (0.9 instead of 1.0)
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output) * 0.9, disc_real_output)

    # Generated loss
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def feature_matching_loss(disc_real_features, disc_fake_features):
    """Feature matching loss for stable training"""
    loss = 0
    for real_feat, fake_feat in zip(disc_real_features, disc_fake_features):
        loss += tf.reduce_mean(tf.abs(real_feat - fake_feat))
    return loss

# Training utilities
class Pix2PixTrainer:
    """Professional Pix2Pix trainer with advanced features"""
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.perceptual_loss_fn = PerceptualLoss()

        # Metrics
        self.gen_total_loss = tf.keras.metrics.Mean()
        self.gen_gan_loss = tf.keras.metrics.Mean()
        self.gen_l1_loss = tf.keras.metrics.Mean()
        self.disc_loss = tf.keras.metrics.Mean()

    @tf.function
    def train_step(self, input_image, target_image):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake image
            gen_output = self.generator(input_image, training=True)

            # Discriminator predictions
            disc_real_output = self.discriminator([input_image, target_image], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            # Calculate losses
            gen_total_loss, gen_gan_loss, gen_l1_loss, gen_perceptual_loss = generator_loss(
                disc_generated_output, gen_output, target_image, self.perceptual_loss_fn)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # Calculate gradients
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        # Update metrics
        self.gen_total_loss.update_state(gen_total_loss)
        self.gen_gan_loss.update_state(gen_gan_loss)
        self.gen_l1_loss.update_state(gen_l1_loss)
        self.disc_loss.update_state(disc_loss)

        return {
            'gen_total_loss': gen_total_loss,
            'gen_gan_loss': gen_gan_loss,
            'gen_l1_loss': gen_l1_loss,
            'gen_perceptual_loss': gen_perceptual_loss,
            'disc_loss': disc_loss
        }
