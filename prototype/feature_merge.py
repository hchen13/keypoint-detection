import tensorflow as tf

from tensorflow.python.keras.layers import Concatenate, UpSampling2D
from prototype.base_layers import ConvBn


class FeatureMerge(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(FeatureMerge, self).__init__(**kwargs)
        self.f = filters

    def build(self, input_shape):
        self.conv1_layer = ConvBn(filters=self.f, kernel_size=1, padding='same', name='conv1x1')
        self.conv3_layer = ConvBn(filters=self.f, kernel_size=3, padding='same', name='conv3x3')
        self.concat_layer = Concatenate(name='concat')
        self.unpool_layer = UpSampling2D(name='unpool')

    def call(self, inputs, **kwargs):
        h, f = inputs
        unpool = self.unpool_layer(h)
        concat = self.concat_layer([unpool, f])
        conv1 = self.conv1_layer(concat)
        conv2 = self.conv3_layer(conv1)
        return conv2