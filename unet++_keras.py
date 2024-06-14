from keras import backend as K
from keras.layers import concatenate, Conv2DTranspose, Activation
from keras.layers import BatchNormalization


from keras.layers import *
from keras.models import *
from keras.optimizers import *

dropout_rate = 0.8

def MFEBBlock(input_tensor, nb_filter, kernel_size=3):
    x = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), padding='same')(input_tensor)
    x = Activation('selu')(x)
    x1 = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), padding='same', dilation=1)(x)
    x1= Activation('selu')(x1)
    x2 = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), padding='same', dilation=2)(x)
    x2= Activation('selu')(x2)
    x3 = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), padding='same', dilation=3)(x)
    x3= Activation('selu')(x3)
    out = keras.layers.Concatenate(axis=1)([x, x1, x2, x3])
    out = Activation('selu')(out)
    out = keras.layers.Add()([out, input_tensor])
    out = Activation('selu')(out)
    return out



def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3):

    x = Conv3D(nb_filter, (kernel_size, kernel_size,kernel_size), padding='same')(input_tensor)
    # x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)

    return x


def model_build_func(input_shape, n_labels, using_deep_supervision=False):

    nb_filter = [32,64,128,256,512]

    # Set image data format to channels first
    global bn_axis

    K.set_image_data_format("channels_last")
    bn_axis = -1
    inputs = Input(shape=input_shape, name='input_image')
    out = MFEBBlock(inputs, nb_filternb_filter[0], kernel_size=3)

    conv1_1 = conv_batchnorm_relu_block(out, nb_filter=nb_filter[0])
    pool1 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[1])
    pool2 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    up1_2 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  nb_filter=nb_filter[0])

    conv3_1 = conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2])
    pool3 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    up2_2 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filter[1])

    up1_3 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filter[0])

    conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[3])
    pool4 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), name='pool4')(conv4_1)

    up3_2 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2])

    up2_3 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[1])

    up1_4 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filter[0])

    conv5_1 = conv_batchnorm_relu_block(pool4, nb_filter=nb_filter[4])

    up4_2 = Conv3DTranspose(nb_filter[3], (2, 2, 2), strides=(2, 2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filter[3])

    up3_3 = Conv3DTranspose(nb_filter[2], (2, 2, 2), strides=(2, 2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filter[2])

    up2_4 = Conv3DTranspose(nb_filter[1], (2, 2, 2), strides=(2, 2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filter[1])

    up1_5 = Conv3DTranspose(nb_filter[0], (2, 2, 2), strides=(2, 2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filter[0])

    nestnet_output_1 = Conv3D(n_labels, (1, 1,1), activation='sigmoid', name='output_1',padding='same')(conv1_2)
    nestnet_output_2 = Conv3D(n_labels, (1, 1,1), activation='sigmoid', name='output_2', padding='same' )(conv1_3)
    nestnet_output_3 = Conv3D(n_labels, (1, 1,1), activation='sigmoid', name='output_3', padding='same')(conv1_4)
    nestnet_output_4 = Conv3D(n_labels, (1, 1,1), activation='sigmoid', name='output_4', padding='same')(conv1_5)

    if using_deep_supervision:
        model = Model(input=inputs, output=[nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4])
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_4)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model
