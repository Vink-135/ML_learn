"""
Data loading and preprocessing for Professional Pix2Pix
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import os

def load_image(image_file):
    """Load and preprocess a single image pair"""
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2

    # Split the image into input and target
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    # Convert to float32
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):
    """Resize images to target size"""
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image, height=256, width=256):
    """Random crop for data augmentation"""
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, height, width, 3])

    return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
    """Normalize images to [-1, 1]"""
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function
def random_jitter(input_image, real_image, height=256, width=256):
    """Apply random jittering for data augmentation"""
    # Resize to slightly larger size
    input_image, real_image = resize(input_image, real_image, height + 30, width + 30)

    # Random crop back to target size
    input_image, real_image = random_crop(input_image, real_image, height, width)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file, height=256, width=256):
    """Load and preprocess training image"""
    input_image, real_image = load_image(image_file)
    input_image, real_image = random_jitter(input_image, real_image, height, width)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file, height=256, width=256):
    """Load and preprocess test image"""
    input_image, real_image = load_image(image_file)
    input_image, real_image = resize(input_image, real_image, height, width)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def get_dataset(path, train=True, batch_size=1, height=256, width=256):
    """Create dataset from path"""
    path = Path(path)

    if train:
        dataset_path = path / "train"
    else:
        dataset_path = path / "val"

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
        # Create dummy dataset for testing
        return create_dummy_dataset(batch_size, height, width)

    # Get list of image files
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))

    if not image_files:
        print(f"Warning: No images found in {dataset_path}")
        # Create dummy dataset for testing
        return create_dummy_dataset(batch_size, height, width)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices([str(f) for f in image_files])

    if train:
        dataset = dataset.map(lambda x: load_image_train(x, height, width),
                            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=400)
    else:
        dataset = dataset.map(lambda x: load_image_test(x, height, width),
                            num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def create_dummy_dataset(batch_size=1, height=256, width=256):
    """Create dummy dataset for testing when no real data is available"""
    print(f"Creating dummy dataset: batch_size={batch_size}, size={height}x{width}")

    def generate_dummy_data():
        # Create semantic-like input (simple colored shapes)
        input_img = tf.random.uniform([height, width, 3], 0, 1)

        # Create realistic-like target (more complex patterns)
        target_img = tf.random.normal([height, width, 3], 0.5, 0.3)
        target_img = tf.clip_by_value(target_img, 0, 1)

        # Normalize to [-1, 1]
        input_img = (input_img * 2) - 1
        target_img = (target_img * 2) - 1

        return input_img, target_img

    dataset = tf.data.Dataset.from_generator(
        generate_dummy_data,
        output_signature=(
            tf.TensorSpec(shape=(height, width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(height, width, 3), dtype=tf.float32)
        )
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()  # Infinite dataset for testing
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset