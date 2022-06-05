import sys

sys.path.append('..')
from utils import *

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import (
            MeanSquaredError, CategoricalCrossentropy
        )
from tensorflow.keras import regularizers

DROPOUT = 0.2
NUM_CHANNELS = 64
REG_CONSTANT = 0.0001

class ResNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args
        self.is_training = True

        # Neural Net
        self.input_boards = tf.keras.Input(shape=[self.board_x, self.board_y])  # s: batch_size x board_x x board_y
        x_image = Reshape([self.board_x, self.board_y, 1])(self.input_boards)                # batch_size  x board_x x board_y x 1
        #x_image = LeakyReLU()(Conv2D(NUM_CHANNELS, kernel_size=(3, 3), strides=(1, 1), name='conv', padding='same', use_bias=False)(self.input_boards))
        x_image = LeakyReLU()(BatchNormalization(axis=-1, name='bn_first')(Conv2D(NUM_CHANNELS, kernel_size=(3, 3),
                                                                                  kernel_regularizer=regularizers.l2(REG_CONSTANT), data_format="channels_last",
                                                                                  activation="linear", name='conv', padding='same', use_bias=False)(x_image)))

        residual_tower = self.residual_block(inputLayer=x_image, kernel_size=3, filters=NUM_CHANNELS, stage=1,
                                                 block='a')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=2, block='b')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=3, block='c')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=4, block='d')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=5, block='e')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=6, block='g')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=7, block='h')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=8, block='i')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=9, block='j')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=10, block='k')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=11, block='m')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=12, block='n')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=13, block='o')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=14, block='p')
        """
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=15, block='q')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=16, block='r')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=17, block='s')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=18, block='t')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=19, block='u')
        residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=NUM_CHANNELS,
                                             stage=20, block='v')
        """

        # Maybe make conv2d filters = channels
        policy = Conv2D(32, kernel_size=(1, 1), data_format="channels_last", activation="linear", name='pi', kernel_regularizer=regularizers.l2(REG_CONSTANT),
                                            padding='same', use_bias=False)(residual_tower)
        policy = BatchNormalization(axis=-1, name='bn_pi')(policy, training=self.is_training)
        policy = LeakyReLU()(policy)
        policy = Flatten(name='p_flatten')(policy)
        self.pi = Activation('softmax', name='policy_head')(Dense(self.action_size, activation="linear", use_bias=False, kernel_regularizer=regularizers.l2(REG_CONSTANT))(policy))

        value = Conv2D(1, kernel_size=(1, 1), data_format="channels_last", activation="linear", kernel_regularizer=regularizers.l2(REG_CONSTANT), name='v',
                                           padding='same', use_bias=False)(residual_tower)
        value = BatchNormalization(axis=-1, name='bn_v')(value, training=self.is_training)
        value = LeakyReLU()(value)
        value = Flatten(name='v_flatten')(value)
        value = Dense(units=256, activation="linear", use_bias=False, kernel_regularizer=regularizers.l2(REG_CONSTANT))(value)
        value = LeakyReLU()(value)
        self.v = Dense(1, activation='tanh', use_bias=False, kernel_regularizer=regularizers.l2(REG_CONSTANT), name='value_head')(value)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss={
                          'value_head': MeanSquaredError(),
                          'policy_head': CategoricalCrossentropy()
                      },
                      loss_weights={'value_head': 10, 'policy_head': 0.4}, optimizer=Adam(args.lr))
        #print(self.model.summary())

    def residual_block(self, inputLayer, filters, kernel_size, stage, block):
        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'

        residual_layer = Conv2D(filters, kernel_size=(kernel_size, kernel_size), kernel_regularizer=regularizers.l2(REG_CONSTANT),
                                data_format="channels_last", activation="linear",
                                                    name=conv_name + '2a', padding='same',
                                                    use_bias=False)(inputLayer)
        residual_layer = BatchNormalization(axis=-1, name=bn_name + '2a')(residual_layer, training=self.is_training)
        residual_layer = LeakyReLU()(residual_layer)
        residual_layer = Conv2D(filters, kernel_size=(kernel_size, kernel_size), kernel_regularizer=regularizers.l2(REG_CONSTANT),
                                data_format="channels_last", activation="linear",
                                                    name=conv_name + '2b', padding='same',
                                                    use_bias=False)(residual_layer)
        residual_layer = BatchNormalization(axis=-1, name=bn_name + '2b')(residual_layer, training=self.is_training)
        add_shortcut = Add()([residual_layer, inputLayer])
        residual_result = LeakyReLU()(add_shortcut)

        return residual_result

