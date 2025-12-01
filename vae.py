#AUTO ENCODERS

# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.datasets import mnist
# import matplotlib.pyplot as plt

# # Load data
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# # Flatten images (28x28 → 784)
# x_train = x_train.reshape(-1, 784)
# x_test = x_test.reshape(-1, 784)

# # Define encoder
# input_img = layers.Input(shape=(784,))
# encoded = layers.Dense(64, activation='relu')(input_img)

# # Define decoder
# decoded = layers.Dense(784, activation='sigmoid')(encoded)

# # Autoencoder model
# autoencoder = models.Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss='mse')

# # Train
# autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))


# <---------------------------------------------------------------------------------------------------------------------------------------------------->


# #VARIATONAL AUTOENCODER

# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np

# # Latent dimension
# latent_dim = 2

# # Encoder
# encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
# x = layers.Flatten()(encoder_inputs)
# x = layers.Dense(128, activation="relu")(x)
# z_mean = layers.Dense(latent_dim)(x)
# z_log_var = layers.Dense(latent_dim)(x)

# # Sampling layer
# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# z = layers.Lambda(sampling)([z_mean, z_log_var])
# encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# # Decoder
# latent_inputs = tf.keras.Input(shape=(latent_dim,))
# x = layers.Dense(128, activation="relu")(latent_inputs)
# x = layers.Dense(28*28, activation="sigmoid")(x)
# decoder_outputs = layers.Reshape((28, 28, 1))(x)
# decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

# # VAE model
# class VAE(tf.keras.Model):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
    
#     def compile(self, optimizer):
#         super().compile()
#         self.optimizer = optimizer
#         self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

#     def train_step(self, data):
#         if isinstance(data, tuple): data = data[0]
#         with tf.GradientTape() as tape:
#             z_mean, z_log_var, z = self.encoder(data)
#             reconstruction = self.decoder(z)
#             reconstruction_loss = tf.reduce_mean(
#                 tf.keras.losses.binary_crossentropy(data, reconstruction)
#             )
#             kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#             total_loss = reconstruction_loss + kl_loss
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         return {"loss": self.total_loss_tracker.result()}



