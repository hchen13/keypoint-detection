import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
from glob import glob
import tensorflow as tf

import cv2
import numpy as np
from ethan_toolbox import draw_points, show_image
from mtcnn import MTCNN
from tqdm import tqdm

from dataset_management.tools import make_train_sample
from dataset_management.tfannotation import create_tfrecord
from settings import ENV

mtcnn = MTCNN()


def extract_keypoints(image, mode='absolute'):
    available_modes = ['absolute', 'relative']
    if mode not in available_modes:
        raise TypeError(f'[extract_keypoints] unknown mode: {mode}, has to be in one of {available_modes}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = mtcnn.detect_faces(image)
    if len(faces) != 1:
        return

    keypoints = np.empty((5, 2))
    face = faces[0]
    for i, kp_name in enumerate(['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']):
        p = face['keypoints'][kp_name]
        keypoints[i, 0] = p[0]
        keypoints[i, 1] = p[1]
    if mode == 'relative':
        shape = np.array([image.shape[1], image.shape[0]])
        keypoints /= shape
    return keypoints


def process_file(file, image_size=256):
    image = cv2.imread(file)
    keypoints = extract_keypoints(image, mode='relative')
    if keypoints is None:
        return None
    image, heatmaps, offsets = make_train_sample(image, keypoints, image_size=image_size)
    serialized = create_tfrecord(image, heatmaps, offsets)
    return serialized


def _create_dataset(files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[info] start creating dataset: \n"
          f"\tno. images: {len(files)}\n"
          f"\tdestination directory: {dest_dir}")
    error_count = 0
    global_count = 0
    for i, file in tqdm(enumerate(files), total=len(files), mininterval=.5):
        serialized = process_file(file)
        if serialized is None:
            error_count += 1
            continue

        filename = f'data{global_count:05d}.tfrecords'
        path = os.path.join(dest_dir, filename)
        with tf.io.TFRecordWriter(path) as writer:
            writer.write(serialized)
        global_count += 1
    print(f"[info] finished, summary: \n"
          f"\terrors: {error_count}"
          f"\trecords created: {global_count}")


def create_dataset(files, dest_dir):
    size = len(files)
    valid_size = min(size // 10, 1000)
    train_files = files[:-valid_size]
    valid_files = files[-valid_size:]
    _create_dataset(train_files, os.path.join(dest_dir, 'train'))
    _create_dataset(valid_files, os.path.join(dest_dir, 'valid'))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    if ENV == 'local':
        image_dirs = [
            os.path.join(home, 'datasets/Selfie-dataset/images'),
            os.path.join(home, 'datasets/genki4k/files')
        ]
        destination_dir = '/Users/ethan/datasets/face-keypoints'
    else:
        image_dirs = [
            os.path.join(home, 'datasets/Selfie-dataset/images'),
            os.path.join(home, 'datasets/genki4k/files'),
        ]
        destination_dir = '/media/ethan/DataStorage/face-keypoints'

    all_files = []
    for folder in image_dirs:
        p = os.path.join(folder, '*.jpg')
        images = glob(p)
        all_files += images

    create_dataset(all_files, destination_dir)