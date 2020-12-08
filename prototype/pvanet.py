import os

import cv2
import tensorflow as tf
import numpy as np
from ethan_toolbox import show_image, draw_points, convert_points_range

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import MaxPooling2D, BatchNormalization, ReLU, Concatenate, Reshape, UpSampling2D, Conv2D, Conv2DTranspose

from prototype.base_layers import conv, ConvBn
from prototype.crelu import CRelu, ResidualCRelu
from prototype.feature_merge import FeatureMerge
from prototype.identity import Identity
from prototype.inception import Inception
from settings import PROJECT_ROOT


class KeypointNet:
    def __init__(self, K, image_size=None):
        self.model = None
        self.K = K
        self.build(input_shape=(image_size, image_size, 3))
        if self.model is not None:
            self.model.predict(np.random.uniform(size=(1, 256, 256, 3)))

    def build(self, input_shape):
        input_tensor = Input(shape=input_shape, name='input_image')
        conv11 = CRelu(kernel_size=7, filters=16, strides=2, name='conv1_1')(input_tensor)
        pool11 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool1_1')(conv11)

        conv21 = ResidualCRelu(params="3 1 PJ 32-24-128 NO", name='conv2_1')(pool11)
        conv22 = ResidualCRelu(params="3 1 NO 32-24-128 BN", name='conv2_2')(conv21)
        conv23 = ResidualCRelu(params="3 1 NO 32-24-128 BN", name='conv2_3')(conv22)

        conv31 = ResidualCRelu(params="3 2 PJ 64-48-128 BN", name='conv3_1')(conv23)
        conv32 = ResidualCRelu(params="3 1 NO 64-48-128 BN", name='conv3_2')(conv31)
        conv33 = ResidualCRelu(params="3 1 PJ 64-48-192 BN", name='conv3_3')(conv32)
        conv34 = ResidualCRelu(params="3 1 NO 64-48-192 BN", name='conv3_4')(conv33)

        conv41 = Inception(params="2 PJ 64 64-128 32-48-48 256", name='conv4_1')(conv34)
        conv42 = Inception(params="1 NO 64 64-128 32-48-48 256", name='conv4_2')(conv41)
        conv43 = Inception(params="1 NO 64 64-128 32-48-48 256", name='conv4_3')(conv42)
        conv44 = Inception(params="1 NO 64 64-128 32-48-48 256", name='conv4_4')(conv43)

        conv51 = Inception(params="2 PJ 64 96-192 32-64-64 384", name='conv5_1')(conv44)
        conv52 = Inception(params="1 NO 64 96-192 32-64-64 384", name='conv5_2')(conv51)
        conv53 = Inception(params="1 NO 64 96-192 32-64-64 384", name='conv5_3')(conv52)
        conv54 = Inception(params="1 NO 64 96-192 32-64-64 384", name='conv5_4_pre')(conv53)
        conv54 = BatchNormalization(scale=False, name='conv5_4_bn')(conv54)
        conv54 = ReLU(name='conv5_4')(conv54)

        merged = FeatureMerge(filters=128, name='merge1')([conv54, conv44])
        merged = FeatureMerge(filters=64, name='merge2')([merged, conv34])
        merged = FeatureMerge(filters=32, name='merge3')([merged, conv23])
        up = Conv2DTranspose(32, 3, strides=2, padding='same', name='upscale1')(merged)
        up = Conv2DTranspose(32, 3, strides=2, padding='same', name='upscale2')(up)

        heatmap = Conv2D(self.K, 3, padding='same', activation='sigmoid', name='heatmap')(up)
        offset = Conv2D(self.K * 2, 3, padding='same', activation=None, name='offset')(up)
        identity = Identity(name='identity_tensor')(up)
        self.model = Model(inputs=input_tensor, outputs=[heatmap, offset, identity], name='pvanet')


    def init_pvanet(self, path=None):
        if path is None:
            path = os.path.join(PROJECT_ROOT, 'weights', 'pvanet_init.h5')
        self.model.load_weights(path, by_name=True)

    def load_weights(self, path_or_name):
        if not os.path.exists(path_or_name):
            path_or_name = os.path.join(PROJECT_ROOT, 'weights', path_or_name)
        if not os.path.exists(path_or_name):
            raise ValueError("[KeypointNet] weights file does not exist.")
        self.model.load_weights(path_or_name, by_name=True)

    def detect(self, image, disk_radius=10, refine_mode='recursive'):
        if refine_mode not in ['recursive', 'mean']:
            raise KeyError(f"[KeypointNet] invalid detection refine algorithm: {refine_mode}")

        input_image = create_feed(image)
        heatmap, offset, identity = self.model.predict(input_image)
        heatmap = heatmap[0]
        offset = offset[0]

        keypoints = []
        for k in range(self.K):
            o = offset[..., 2 * k : 2 * k + 2]
            point_map = identity - o * disk_radius
            points = np.reshape(point_map, (-1, 2))
            mask = heatmap[..., k]
            if np.max(mask) < .5:
                keypoints.append(np.array([-1, -1]))
                continue
            if refine_mode == 'mean':
                rank = np.argsort(mask, axis=None)[::-1]
                candidate_ids = rank[:5]
                candidates = points[candidate_ids]
                point = np.mean(candidates, axis=0)
                keypoints.append(point)
            else:
                y, x = np.where(mask == np.max(mask))
                candidate = [y[0], x[0]]
                for _ in range(4):
                    y, x = candidate
                    refined = point_map[int(y), int(x)]
                    candidate = refined[1], refined[0]
                keypoints.append(refined)
        keypoints = np.array(keypoints)
        return convert_points_range(keypoints, (256, 256), mode='absolute2relative')


def create_feed(image_or_path):
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise ValueError(f"Invalid type: {type(image_or_path)}")
    feed = cv2.resize(image, (256, 256))
    feed = np.expand_dims(feed, axis=0)
    return feed / 127.5 - 1.


if __name__ == '__main__':
    import settings

    net = KeypointNet(5)
    # net.init_pvanet(os.path.join(PROJECT_ROOT, 'weights', 'pvanet_init.h5'))
    net.model.load_weights(os.path.join(PROJECT_ROOT, 'weights', 'test.h5'))
    # net.model.summary()
    image = cv2.imread(os.path.join(settings.PROJECT_ROOT, 'tests', '1.jpg'))
    keypoints = net.detect(image)