from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Add, Concatenate, Conv2D, Dropout, \
                         Input, PReLU, Reshape, UpSampling2D
from keras.regularizers import l2


def build_block(filters, kernel_size):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=(1,1), padding='same', use_bias=True,
                      kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4),
                      bias_initializer='zeros')(input)
        prelu = PReLU()(conv)
        dropout = Dropout(rate=0.8)(prelu)
        return dropout

    return f

def build_model(input_shape):
    input = Input(shape=input_shape)

    f1 = build_block(filters=96, kernel_size=(3,3))(input)
    f2 = build_block(filters=76, kernel_size=(3,3))(f1)
    f3 = build_block(filters=65, kernel_size=(3,3))(f2)
    f4 = build_block(filters=55, kernel_size=(3,3))(f3)
    f5 = build_block(filters=47, kernel_size=(3,3))(f4)
    f6 = build_block(filters=39, kernel_size=(3,3))(f5)
    f7 = build_block(filters=32, kernel_size=(3,3))(f6)

    f_out = Concatenate()([f1, f2, f3, f4, f5, f6, f7])

    a1 = build_block(filters=64, kernel_size=(1,1))(f_out)
    b1 = build_block(filters=32, kernel_size=(1,1))(f_out)
    b2 = build_block(filters=32, kernel_size=(1,1))(b1)

    r_out = Concatenate()([a1, b2])
    net_out = Conv2D(filters=16, kernel_size=(1,1),
                     strides=(1,1), padding='same', use_bias=True,
                     kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4),
                     bias_initializer='zeros')(r_out)

    net_reshaped = Reshape((-1, input_shape[1] * 4, 1))(net_out)
    
    upsampled = UpSampling2D(size=(4,4), interpolation='bilinear')(input)

    output = Add()([net_reshaped, upsampled])

    model = Model(inputs=input, outputs=output)
    return model
