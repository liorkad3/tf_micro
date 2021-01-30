import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa


def WnConv2d(*args, **kwargs):
    return tfa.layers.WeightNormalization(layers.Conv2D(*args, **kwargs), data_init=True)

def WnDense(*args, **kwargs):
    return tfa.layers.WeightNormalization(layers.Dense(*args, **kwargs), data_init=True)


def ConvBN2d(*args, **kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(*args, **kwargs),
        tf.keras.layers.BatchNormalization()
    ])

def DenseBN(*args, **kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(*args, **kwargs),
        tf.keras.layers.BatchNormalization()
    ]) 

class DownSample(tf.keras.Model):
    def __init__(self, channels, filt_size=3, stride=2):
        super(DownSample, self).__init__()
        
        self.stride = stride
        assert filt_size == 3

        a = tf.constant([1., 2., 1.])
        filt = (a[:, None] * a[None, :])
        filt = filt / tf.math.reduce_sum(filt)

        self.filt = tf.repeat(filt[:, :, None, None], repeats=channels, axis=2)

        self.strides = [1, self.stride, self.stride, 1]
        self.paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        

    def call(self, x):
        x = tf.pad(x, self.paddings, mode='REFLECT')
        return tf.nn.depthwise_conv2d(x, self.filt, self.strides, padding='VALID')

if __name__ == "__main__":
    # input_shape = (4, 32, 80, 30)
    # x = tf.random.normal(input_shape)
    # y = wn_conv2d(x, 64, (1,1), stride=2)
    # print(y.shape)
    m = DownSample(32)
    input_shape = (2, 32, 80, 32)
    x = tf.random.normal(input_shape)

    y = m(x)

    print(y.shape)