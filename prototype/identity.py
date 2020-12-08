import tensorflow as tf
from tensorflow.python.framework import tensor_shape


class Identity(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]
        x = tf.linspace(0., tf.cast(width, tf.float32) - 1., width)
        y = tf.linspace(0., tf.cast(height, tf.float32) - 1., height)
        x, y = tf.meshgrid(x, y)
        eye = tf.stack([x, y], axis=2)
        return eye

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_size, height, width, channels = [input_shape[i] for i in range(4)]
        output_shape = (batch_size, input_shape[1], input_shape[2], 2)
        return output_shape


if __name__ == '__main__':
    tensor = tf.keras.Input(shape=(16, 16, 10))
    i = Identity()(tensor)
    print(i.shape)
