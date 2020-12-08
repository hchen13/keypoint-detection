import os
import random
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from ethan_toolbox import convert_points_range
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.ops import random_ops, math_ops, control_flow_ops

from dataset_management.tfannotation import read_tfrecord


def create_identity_tensor(size):
    x = np.linspace(0., size - 1, size)
    y = np.linspace(0., size - 1, size)
    x, y = np.meshgrid(x, y)
    tensor = np.stack([x, y], axis=2)
    return tensor


def make_train_sample(image, keypoints, image_size, disk_radius=10):
    image = cv2.resize(image, (image_size, image_size))
    K = keypoints.shape[0]
    identity = create_identity_tensor(image_size)
    heatmaps, offsets = [], []
    for k in range(K):
        keypoint = keypoints[k]
        keypoint = convert_points_range(keypoint, image.shape, mode='relative2absolute')
        offset = identity - keypoint
        dist = np.sqrt(np.sum(offset ** 2, axis=2))
        heatmap = (dist <= disk_radius).astype('float32')
        heatmaps.append(heatmap)
        offset = offset * np.expand_dims(heatmap, axis=2) / disk_radius
        offsets.append(offset)

    heatmaps = np.stack(heatmaps, axis=2)
    offsets = np.concatenate(offsets, axis=2).astype('float32')
    return image, heatmaps, offsets


def load_dataset(dataset_dir, batch_size, augment=None, shuffle=True):
    print("[info] loading training dataset from tfrecord files...", flush=True)
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    train_images, valid_images = [glob(os.path.join(p, '*.tfrecords')) for p in [train_dir, valid_dir]]
    if shuffle:
        random.shuffle(train_images)
    raw = tf.data.TFRecordDataset(train_images)
    trainset = raw.map(read_tfrecord)
    raw = tf.data.TFRecordDataset(valid_images)
    validset = raw.map(read_tfrecord)
    if augment is not None:
        trainset = augment(trainset).batch(batch_size).prefetch(AUTOTUNE)
    else:
        trainset = trainset.batch(batch_size).prefetch(AUTOTUNE)
    validset = validset.batch(batch_size).prefetch(AUTOTUNE)
    print(f'[info] dataset preparation complete, '
          f'training size: {len(train_images)}, validation size: {len(valid_images)}.\n'
          f'batch size: {batch_size}.\n', flush=True)
    return trainset, validset


def horizontal_flip(image, heatmap, offset):
    image = image[:, ::-1, :]

    heatmap = heatmap[:, ::-1, :]
    channels = tf.unstack(heatmap, num=5, axis=-1)
    heatmap = tf.stack([channels[1], channels[0], channels[2], channels[4], channels[3]], axis=-1)

    offset = offset[:, ::-1, :]
    flip_tensor = tf.constant([[[-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]]], dtype=tf.float32)
    offset = tf.multiply(offset, flip_tensor)
    channels = tf.unstack(offset, num=10, axis=-1)
    offset = tf.stack([
        channels[2], channels[3],
        channels[0], channels[1],
        channels[4], channels[5],
        channels[8], channels[9],
        channels[6], channels[7]
    ], axis=-1)

    return image, heatmap, offset


def random_horizontal_flip(image, heatmap, offset):
    flag = random_ops.random_uniform([], 0, 1.)
    flag = math_ops.less(flag, .5)
    results = tf.cond(
        flag,
        lambda : horizontal_flip(image, heatmap, offset),
        lambda : (image, heatmap, offset)
    )
    return results

def augment(dataset):
    def _transform(image, heatmap, offset):
        x = tf.image.random_brightness(image, max_delta=.3)
        x = tf.image.random_saturation(x, .8, 1.2)
        x = tf.image.random_contrast(x, .8, 1.2)
        x = tf.image.random_hue(x, max_delta=.1)
        x, heatmap, offset = random_horizontal_flip(x, heatmap, offset)
        return x, heatmap, offset
    return dataset.map(_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
