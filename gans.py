# # # import tensorflow as tf
# # # from tensorflow.keras import layers
# # # import numpy as np

# # # # Load MNIST digits
# # # (x_train, _), _ = tf.keras.datasets.mnist.load_data()
# # # x_train = x_train / 127.5 - 1.0  # Normalize to [-1, 1]
# # # x_train = x_train.reshape((-1, 28, 28, 1))


# # # #BULD THE GENERATOR
# # # def build_generator():
# # #     model = tf.keras.Sequential([
# # #         layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
# # #         layers.BatchNormalization(),
# # #         layers.LeakyReLU(),

# # #         layers.Reshape((7, 7, 256)),

# # #         layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
# # #         layers.BatchNormalization(),
# # #         layers.LeakyReLU(),

# # #         layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
# # #         layers.BatchNormalization(),
# # #         layers.LeakyReLU(),

# # #         layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
# # #     ])
# # #     return model


# # # #BUILD THE DISCRIMINATOR

# # # def build_discriminator():
# # #     model = tf.keras.Sequential([
# # #         layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
# # #                       input_shape=[28, 28, 1]),
# # #         layers.LeakyReLU(),
# # #         layers.Dropout(0.3),

# # #         layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
# # #         layers.LeakyReLU(),
# # #         layers.Dropout(0.3),

# # #         layers.Flatten(),
# # #         layers.Dense(1)
# # #     ])
# # #     return model


# # # #DEFINE LOSS AND OPTIMIZER

# # # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# # # def discriminator_loss(real_output, fake_output):
# # #     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
# # #     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
# # #     return real_loss + fake_loss

# # # def generator_loss(fake_output):
# # #     return cross_entropy(tf.ones_like(fake_output), fake_output)

# # # # Optimizers
# # # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# # # discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# # # #TRANING THE LOOP

# # # @tf.function
# # # def train_step(images, generator, discriminator):
# # #     noise = tf.random.normal([BATCH_SIZE, 100])

# # #     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
# # #         generated_images = generator(noise, training=True)

# # #         real_output = discriminator(images, training=True)
# # #         fake_output = discriminator(generated_images, training=True)

# # #         gen_loss = generator_loss(fake_output)
# # #         disc_loss = discriminator_loss(real_output, fake_output)

# # #     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
# # #     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

# # #     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
# # #     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# # #     return gen_loss, disc_loss


# # # #FULL TRAINING
# # # EPOCHS = 50
# # # BATCH_SIZE = 128
# # # BUFFER_SIZE = 60000

# # # # Prepare dataset
# # # train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# # # # Build models
# # # generator = build_generator()
# # # discriminator = build_discriminator()

# # # for epoch in range(EPOCHS):
# # #     for image_batch in train_dataset:
# # #         g_loss, d_loss = train_step(image_batch, generator, discriminator)

# # #     print(f"Epoch {epoch+1}, Generator loss: {g_loss.numpy():.4f}, Discriminator loss: {d_loss.numpy():.4f}")





# # # #SAMPLE IMAGES
# # # # 
# # # import matplotlib.pyplot as plt

# # # def generate_images(model, epoch):
# # #     noise = tf.random.normal([16, 100])
# # #     generated = model(noise, training=False)

# # #     fig = plt.figure(figsize=(4, 4))
# # #     for i in range(generated.shape[0]):
# # #         plt.subplot(4, 4, i+1)
# # #         plt.imshow(generated[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
# # #         plt.axis('off')
# # #     plt.suptitle(f"Epoch {epoch}")
# # #     plt.show()

# # # generate_images(generator, EPOCHS)    



# # # <---------------------------------------------------------------------------------------------------------------------------------------------------->


# # #DC-GAN

# # #DATA PREP

# # # import tensorflow as tf
# # # from tensorflow.keras import layers
# # # import matplotlib.pyplot as plt
# # # import numpy as np

# # # # Load and normalize dataset
# # # (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
# # # x_train = x_train.astype("float32")
# # # x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
# # # x_train = x_train.reshape((-1, 28, 28, 1))

# # # BUFFER_SIZE = 60000
# # # BATCH_SIZE = 128


# # # #BUILD THE GENERATOR

# # # def build_generator():
# # #     model = tf.keras.Sequential([
# # #         layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
# # #         layers.BatchNormalization(),
# # #         layers.ReLU(),

# # #         layers.Reshape((7, 7, 256)),

# # #         layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),
# # #         layers.BatchNormalization(),
# # #         layers.ReLU(),

# # #         layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
# # #         layers.BatchNormalization(),
# # #         layers.ReLU(),

# # #         layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")
# # #     ])
# # #     return model


# # # #BUILD THE DISCRIMINATOR

# # # def build_discriminator():
# # #     model = tf.keras.Sequential([
# # #         layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]),
# # #         layers.LeakyReLU(),
# # #         layers.Dropout(0.3),

# # #         layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
# # #         layers.LeakyReLU(),
# # #         layers.Dropout(0.3),

# # #         layers.Flatten(),
# # #         layers.Dense(1)
# # #     ])
# # #     return model


# # # #LOSS AND OPTIMIZER

# # # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# # # def discriminator_loss(real_output, fake_output):
# # #     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
# # #     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
# # #     return real_loss + fake_loss

# # # def generator_loss(fake_output):
# # #     return cross_entropy(tf.ones_like(fake_output), fake_output)

# # # generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# # # discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# # # #TRANING FUNCTIONS

# # # @tf.function
# # # def train_step(images, generator, discriminator):
# # #     noise = tf.random.normal([BATCH_SIZE, 100])

# # #     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
# # #         generated_images = generator(noise, training=True)

# # #         real_output = discriminator(images, training=True)
# # #         fake_output = discriminator(generated_images, training=True)

# # #         gen_loss = generator_loss(fake_output)
# # #         disc_loss = discriminator_loss(real_output, fake_output)

# # #     gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
# # #     gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

# # #     generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
# # #     discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

# # #     return gen_loss, disc_loss

# # # #TRANING THE DCGAN


# # # def train(dataset, epochs):
# # #     for epoch in range(epochs):
# # #         for image_batch in dataset:
# # #             g_loss, d_loss = train_step(image_batch, generator, discriminator)
# # #         print(f"Epoch {epoch+1}: Generator Loss = {g_loss.numpy():.4f}, Discriminator Loss = {d_loss.numpy():.4f}")
# # #         generate_and_save_images(generator, epoch+1)

# # # def generate_and_save_images(model, epoch):
# # #     noise = tf.random.normal([16, 100])
# # #     generated_images = model(noise, training=False)

# # #     fig = plt.figure(figsize=(4, 4))
# # #     for i in range(generated_images.shape[0]):
# # #         plt.subplot(4, 4, i + 1)
# # #         plt.imshow((generated_images[i, :, :, 0] * 127.5 + 127.5).numpy().astype("uint8"), cmap="gray")
# # #         plt.axis("off")
# # #     plt.suptitle(f"DCGAN - Epoch {epoch}")
# # #     plt.show()

# # # train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# # # generator = build_generator()
# # # discriminator = build_discriminator()

# # # train(train_dataset, epochs=30)



# # # <---------------------------------------------------------------------------------------------------------------------------------------------------->


# # #LOAD IMAGE AND PREPARE DATASET
# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from tensorflow.keras.preprocessing.image import img_to_array, load_img

# # def load_image(path, scale=4):
# #     hr = load_img(path)
# #     hr = img_to_array(hr)
# #     hr = cv2.resize(hr, (128, 128))  # Crop to square

# #     lr = cv2.resize(hr, (32, 32), interpolation=cv2.INTER_CUBIC)  # Downscale
# #     lr_upscaled = cv2.resize(lr, (128, 128), interpolation=cv2.INTER_CUBIC)  # Bicubic

# #     return lr / 255.0, hr / 255.0, lr_upscaled / 255.0

# # lr, hr, bicubic = load_image("sample.jpg")


# # #BUILD SRGAN GENERATOR

# # from tensorflow.keras import layers, Model

# # def build_generator():
# #     inputs = layers.Input(shape=(32, 32, 3))

# #     x = layers.Conv2D(64, 9, padding='same')(inputs)
# #     x = layers.PReLU(shared_axes=[1, 2])(x)

# #     skip = x

# #     for _ in range(5):
# #         res = layers.Conv2D(64, 3, padding='same')(x)
# #         res = layers.BatchNormalization()(res)
# #         res = layers.PReLU(shared_axes=[1, 2])(res)
# #         res = layers.Conv2D(64, 3, padding='same')(res)
# #         res = layers.BatchNormalization()(res)
# #         x = layers.add([x, res])

# #     x = layers.add([x, skip])
# #     x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
# #     x = layers.PReLU(shared_axes=[1, 2])(x)
# #     x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
# #     x = layers.PReLU(shared_axes=[1, 2])(x)
# #     x = layers.Conv2D(3, 9, padding='same', activation='tanh')(x)

# #     return Model(inputs, x)

# # generator = build_generator()
# # generator.summary()



# # #DISCRIMINATOR (BASED ON CNN)

# # def build_discriminator():
# #     input_img = layers.Input(shape=(128, 128, 3))
# #     x = layers.Conv2D(64, 3, strides=1, padding="same")(input_img)
# #     x = layers.LeakyReLU(0.2)(x)

# #     for filters in [64, 128, 128, 256, 256, 512]:
# #         x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(0.2)(x)

# #     x = layers.Flatten()(x)
# #     x = layers.Dense(1024)(x)
# #     x = layers.LeakyReLU(0.2)(x)
# #     out = layers.Dense(1, activation="sigmoid")(x)

# #     return Model(input_img, out)

# # discriminator = build_discriminator()


# # #GENERATE AN RESOLVED OUTPUT


# # # Upscale LR to HR using generator
# # sr = generator.predict(lr[np.newaxis, ...])
# # sr = sr[0]

# # # Plot
# # plt.figure(figsize=(12, 4))
# # titles = ['Low-Res', 'Bicubic Upscale', 'SRGAN Output', 'Original HR']
# # images = [lr, bicubic, sr, hr]

# # for i in range(4):
# #     plt.subplot(1, 4, i+1)
# #     plt.imshow(images[i])
# #     plt.title(titles[i])
# #     plt.axis("off")

# # plt.tight_layout()
# # plt.show()




# # <---------------------------------------------------------------------------------------------------------------------------------------------------->


# #PIX-2-PIX


# # import tensorflow_datasets as tf

# # # Load facades dataset
# # dataset, info = tfds.load('facades', with_info=True, as_supervised=True)
# # train, test = dataset['train'], dataset['test']


# # #PRE-PROCESSING

# # def preprocess(image_input, image_target):
# #     image_input = tf.image.resize(image_input, [256, 256])
# #     image_target = tf.image.resize(image_target, [256, 256])
# #     image_input = (tf.cast(image_input, tf.float32) / 127.5) - 1
# #     image_target = (tf.cast(image_target, tf.float32) / 127.5) - 1
# #     return image_input, image_target

# # train = train.map(preprocess).batch(1)


# # #BUILD GENERATOR

# # from tensorflow.keras import layers

# # def downsample(filters, size, apply_batchnorm=True):
# #     initializer = tf.random_normal_initializer(0., 0.02)
# #     result = tf.keras.Sequential([
# #         layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
# #     ])
# #     if apply_batchnorm:
# #         result.add(layers.BatchNormalization())
# #     result.add(layers.LeakyReLU())
# #     return result

# # def upsample(filters, size, apply_dropout=False):
# #     initializer = tf.random_normal_initializer(0., 0.02)
# #     result = tf.keras.Sequential([
# #         layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
# #         layers.BatchNormalization()
# #     ])
# #     if apply_dropout:
# #         result.add(layers.Dropout(0.5))
# #     result.add(layers.ReLU())
# #     return result

# # def Generator():
# #     inputs = layers.Input(shape=[256, 256, 3])

# #     down_stack = [
# #         downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
# #         downsample(128, 4),
# #         downsample(256, 4),
# #         downsample(512, 4),
# #         downsample(512, 4),
# #         downsample(512, 4),
# #         downsample(512, 4),
# #         downsample(512, 4), # (bs, 1, 1, 512)
# #     ]

# #     up_stack = [
# #         upsample(512, 4, apply_dropout=True),
# #         upsample(512, 4, apply_dropout=True),
# #         upsample(512, 4, apply_dropout=True),
# #         upsample(512, 4),
# #         upsample(256, 4),
# #         upsample(128, 4),
# #         upsample(64, 4),
# #     ]

# #     x = inputs
# #     skips = []
# #     for down in down_stack:
# #         x = down(x)
# #         skips.append(x)
# #     skips = reversed(skips[:-1])

# #     for up, skip in zip(up_stack, skips):
# #         x = up(x)
# #         x = layers.Concatenate()([x, skip])

# #     last = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
# #     x = last(x)

# #     return tf.keras.Model(inputs=inputs, outputs=x)



# # #DISCRIMINATOR(PATCHGAN)


# # def Discriminator():
# #     initializer = tf.random_normal_initializer(0., 0.02)
# #     inp = layers.Input(shape=[256, 256, 3], name='input_image')
# #     tar = layers.Input(shape=[256, 256, 3], name='target_image')
# #     x = layers.concatenate([inp, tar])  # condition GAN

# #     down1 = downsample(64, 4, False)(x)
# #     down2 = downsample(128, 4)(down1)
# #     down3 = downsample(256, 4)(down2)

# #     zero_pad1 = layers.ZeroPadding2D()(down3)
# #     conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
# #     batchnorm = layers.BatchNormalization()(conv)
# #     leaky_relu = layers.LeakyReLU()(batchnorm)

# #     zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
# #     last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

# #     return tf.keras.Model(inputs=[inp, tar], outputs=last)



# import tensorflow as tf
# import os
# import time
# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# # The original pix2pix paper uses a custom dataset, which we will download.
# _URL = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz'
# path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)

# # Corrected PATH construction to robustly find the train/test directories.
# base_dir = os.path.join(os.path.dirname(path_to_zip), 'facades')
# PATH = os.path.join(base_dir, 'train')
# TEST_PATH = os.path.join(base_dir, 'test')


# BUFFER_SIZE = 400
# BATCH_SIZE = 1
# IMG_WIDTH = 256
# IMG_HEIGHT = 256

# # The downloaded images are combined (input and target). We need to split them.
# def load(image_file):
#     image = tf.io.read_file(image_file)
#     image = tf.image.decode_jpeg(image)

#     w = tf.shape(image)[1]
#     w = w // 2
#     real_image = image[:, :w, :]
#     input_image = image[:, w:, :]

#     input_image = tf.cast(input_image, tf.float32)
#     real_image = tf.cast(real_image, tf.float32)

#     return input_image, real_image

# # Preprocessing: resize, random jitter (augmentation), and normalize
# def resize(input_image, real_image, height, width):
#     input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     return input_image, real_image

# def random_crop(input_image, real_image):
#     stacked_image = tf.stack([input_image, real_image], axis=0)
#     cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
#     return cropped_image[0], cropped_image[1]

# def normalize(input_image, real_image):
#     input_image = (input_image / 127.5) - 1
#     real_image = (real_image / 127.5) - 1
#     return input_image, real_image

# @tf.function()
# def random_jitter(input_image, real_image):
#     # Resizing to 286x286
#     input_image, real_image = resize(input_image, real_image, 286, 286)
#     # Random cropping back to 256x256
#     input_image, real_image = random_crop(input_image, real_image)
#     # Random mirroring
#     if tf.random.uniform(()) > 0.5:
#         input_image = tf.image.flip_left_right(input_image)
#         real_image = tf.image.flip_left_right(real_image)
#     return input_image, real_image

# # Data loading functions for training and testing
# def load_image_train(image_file):
#     input_image, real_image = load(image_file)
#     input_image, real_image = random_jitter(input_image, real_image)
#     input_image, real_image = normalize(input_image, real_image)
#     return input_image, real_image

# def load_image_test(image_file):
#     input_image, real_image = load(image_file)
#     input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
#     input_image, real_image = normalize(input_image, real_image)
#     return input_image, real_image

# # Create the data pipelines
# train_dataset = tf.data.Dataset.list_files(PATH + '/*.jpg')
# train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# test_dataset = tf.data.Dataset.list_files(TEST_PATH + '/*.jpg')
# test_dataset = test_dataset.map(load_image_test)
# test_dataset = test_dataset.batch(BATCH_SIZE)


# # The Generator and Discriminator models remain the same as your code.
# # I'm including them here for completeness.

# # Note on "reportMissingImports" from Pylance/VSCode:
# # If your editor shows an error like "Import tensorflow.keras could not be resolved",
# # it's an issue with the linter's environment configuration and does not affect
# # the script's execution. The import `from tensorflow.keras import layers` is correct.
# from tensorflow.keras import layers

# def downsample(filters, size, apply_batchnorm=True):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     result = tf.keras.Sequential()
#     result.add(
#         layers.Conv2D(filters, size, strides=2, padding='same',
#                       kernel_initializer=initializer, use_bias=False))
#     if apply_batchnorm:
#         result.add(layers.BatchNormalization())
#     result.add(layers.LeakyReLU())
#     return result

# def upsample(filters, size, apply_dropout=False):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     result = tf.keras.Sequential()
#     result.add(
#         layers.Conv2DTranspose(filters, size, strides=2,
#                                 padding='same',
#                                 kernel_initializer=initializer,
#                                 use_bias=False))
#     result.add(layers.BatchNormalization())
#     if apply_dropout:
#         result.add(layers.Dropout(0.5))
#     result.add(layers.ReLU())
#     return result

# def Generator():
#     inputs = layers.Input(shape=[256, 256, 3])
#     down_stack = [
#         downsample(64, 4, apply_batchnorm=False),
#         downsample(128, 4),
#         downsample(256, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#         downsample(512, 4),
#     ]
#     up_stack = [
#         upsample(512, 4, apply_dropout=True),
#         upsample(512, 4, apply_dropout=True),
#         upsample(512, 4, apply_dropout=True),
#         upsample(512, 4),
#         upsample(256, 4),
#         upsample(128, 4),
#         upsample(64, 4),
#     ]
#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
#     x = inputs
#     skips = []
#     for down in down_stack:
#         x = down(x)
#         skips.append(x)
#     skips = reversed(skips[:-1])
#     concat = layers.Concatenate()
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         x = concat([x, skip])
#     x = last(x)
#     return tf.keras.Model(inputs=inputs, outputs=x)

# def Discriminator():
#     initializer = tf.random_normal_initializer(0., 0.02)
#     inp = layers.Input(shape=[256, 256, 3], name='input_image')
#     tar = layers.Input(shape=[256, 256, 3], name='target_image')
#     x = layers.concatenate([inp, tar])
#     down1 = downsample(64, 4, False)(x)
#     down2 = downsample(128, 4)(down1)
#     down3 = downsample(256, 4)(down2)
#     zero_pad1 = layers.ZeroPadding2D()(down3)
#     conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
#     batchnorm = layers.BatchNormalization()(conv)
#     leaky_relu = layers.LeakyReLU()(batchnorm)
#     zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
#     last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
#     return tf.keras.Model(inputs=[inp, tar], outputs=last)

# # Instantiate the models
# generator = Generator()
# discriminator = Discriminator()

# # Define Loss functions and Optimizers
# LAMBDA = 100
# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def discriminator_loss(disc_real_output, disc_generated_output):
#     real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
#     generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
#     total_disc_loss = real_loss + generated_loss
#     return total_disc_loss

# def generator_loss(disc_generated_output, gen_output, target):
#     gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
#     l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
#     total_gen_loss = gan_loss + (LAMBDA * l1_loss)
#     return total_gen_loss, gan_loss, l1_loss

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# # Checkpointing
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# # Function to generate and display images during training
# def generate_images(model, test_input, tar):
#     prediction = model(test_input, training=True)
#     plt.figure(figsize=(15, 15))

#     display_list = [test_input[0], tar[0], prediction[0]]
#     title = ['Input Image', 'Ground Truth', 'Predicted Image']

#     for i in range(3):
#         plt.subplot(1, 3, i+1)
#         plt.title(title[i])
#         # getting the pixel values between [0, 1] to plot it.
#         plt.imshow(display_list[i] * 0.5 + 0.5)
#         plt.axis('off')
#     plt.show()

# # Training Step
# @tf.function
# def train_step(input_image, target, epoch):
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         gen_output = generator(input_image, training=True)

#         disc_real_output = discriminator([input_image, target], training=True)
#         disc_generated_output = discriminator([input_image, gen_output], training=True)

#         gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
#         disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

#     generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
#     discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

# # Full Training Loop
# def fit(train_ds, epochs, test_ds):
#     for epoch in range(epochs):
#         start = time.time()

#         # Display a sample result
#         for example_input, example_target in test_ds.take(1):
#             generate_images(generator, example_input, example_target)
#         print("Epoch: ", epoch)

#         # Train
#         for n, (input_image, target) in train_ds.enumerate():
#             print('.', end='')
#             if (n+1) % 100 == 0:
#                 print()
#             train_step(input_image, target, epoch)
#         print()

#         # saving (checkpoint) the model every 20 epochs
#         if (epoch + 1) % 20 == 0:
#             checkpoint.save(file_prefix=checkpoint_prefix)

#         print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
#     checkpoint.save(file_prefix=checkpoint_prefix)

# # Set epochs and start training
# EPOCHS = 150
# # To run the training, uncomment the line below
# # fit(train_dataset, EPOCHS, test_dataset)

# print("Training script is ready. Uncomment the last line to start training.")



# # <---------------------------------------------------------------------------------------------------------------------------------------------------->



