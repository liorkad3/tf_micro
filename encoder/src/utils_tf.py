import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa


def wn_conv2d(x, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
 prefix='wn_conv'):

    out = tfa.layers.WeightNormalization(layers.Conv2D(
            out_channels, kernel_size, strides=stride, padding='same',
            dilation_rate=dilation, groups=groups, use_bias=False))(x)
    return out
    

def WnConv2d(*args, **kwargs):
    return tfa.layers.WeightNormalization(layers.Conv2D(*args, **kwargs))

def WnDense(*args, **kwargs):
    return tfa.layers.WeightNormalization(layers.Dense(*args, **kwargs))


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


if __name__ == "__main__":
    input_shape = (4, 32, 80, 30)
    x = tf.random.normal(input_shape)
    y = wn_conv2d(x, 64, (1,1), stride=2)
    print(y.shape)