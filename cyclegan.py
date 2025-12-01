import tensorflow_datasets as tfds

dataset, info = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)
train_horses, train_zebras = dataset['trainA'], dataset['trainB']


def preprocess_img(image, label):
    image = tf.image.resize(image, [256, 256])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

train_horses = train_horses.map(preprocess_img).batch(1)
train_zebras = train_zebras.map(preprocess_img).batch(1)


