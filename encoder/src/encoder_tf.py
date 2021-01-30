import tensorflow as tf
from src.utils_tf import WnConv2d, WnDense, ConvBN2d, DenseBN, DownSample

class Encoder(tf.keras.Model):

    def __init__(self, num_speakers, spk_dim=128):
        super(Encoder, self).__init__()
        self.num_speakers = num_speakers        
        self.spk_dim = spk_dim
        ngf = 32

        self.conv_stack = tf.keras.Sequential([
            WnConv2d(ngf, (9, 5), strides=1),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            WnConv2d(ngf, (9, 5), strides=1, dilation_rate=2),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            WnConv2d(ngf, (9, 5), strides=1, dilation_rate=3),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        self.conv_down = tf.keras.Sequential([
            WnConv2d(ngf*2, 3, strides=1, padding='same'),
            DownSample(ngf*2, filt_size=3, stride=2),
            WnConv2d(ngf*4, 3, strides=1, padding='same'),
            DownSample(ngf*4, filt_size=3, stride=2),
            WnConv2d(spk_dim, (3, 1), strides=1),
        ])

        self.dense = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            WnDense(spk_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            WnDense(spk_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            WnDense(spk_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            WnDense(spk_dim)
        ])

        self.fc = tf.keras.layers.Dense(num_speakers)


    def call(self, x):
        x = tf.expand_dims(x, axis=3)
        y = self.conv_stack(x)
        y = self.conv_down(y)
        emb_spk = self.dense(y)
        emb_spk = tf.linalg.normalize(emb_spk, axis=1)[0]
        log_p_s_x = self.fc(emb_spk)
        return emb_spk, log_p_s_x


class Encoder2(tf.keras.Model):

    def __init__(self, num_speakers, spk_dim=128):
        super(Encoder2, self).__init__()
        self.num_speakers = num_speakers        
        self.spk_dim = spk_dim
        ngf = 32

        self.conv_stack = tf.keras.Sequential([
            ConvBN2d(ngf, (9, 5), strides=1),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            ConvBN2d(ngf, (9, 5), strides=1, dilation_rate=2),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            ConvBN2d(ngf, (9, 5), strides=1, dilation_rate=3),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            # tf.keras.layers.Conv2D(ngf, (9, 5), dilation_rate=2 strides=1, use_bias=False)
        ])

        self.conv_down = tf.keras.Sequential([
            ConvBN2d(ngf*2, 3, strides=1, padding='same'),
            DownSample(ngf*2, filt_size=3, stride=2),

            ConvBN2d(ngf*4, 3, strides=1, padding='same'),
            DownSample(ngf*4, filt_size=3, stride=2),

            ConvBN2d(spk_dim, (3, 1), strides=1),
        ])

        self.dense = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            DenseBN(spk_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            DenseBN(spk_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            DenseBN(spk_dim),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            DenseBN(spk_dim)
        ])

        self.fc = tf.keras.layers.Dense(num_speakers)


    def call(self, x):
        # x = tf.expand_dims(x, axis=3)
        y = self.conv_stack(x)
        y = self.conv_down(y)
        emb_spk = self.dense(y)
        # emb_spk = tf.math.l2_normalize(emb_spk, axis=1)
        # emb_spk = tf.linalg.normalize(emb_spk, axis=1)[0]
        log_p_s_x = self.fc(emb_spk)
        return emb_spk, log_p_s_x


if __name__ == "__main__":
    input_shape = (4, 80, 32)
    x = tf.random.normal(input_shape)

    model = Encoder2(2)
    e, s = model(x)
    print(s.shape, e.shape)

    # model.summary()
    # model.save('/home/liork/Downloads/TfMicro/encoder/models/tf_model')
    # x = tf.constant([[1.0, -6, 7, 4], [2, 6, -12, 0]])
    # y = tf.linalg.normalize(x, axis=1)[0]
    # print(y.shape)
    # print(y)
