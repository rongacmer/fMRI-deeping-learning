# FCN
from basenet import *
import tensorflow as tf
import numpy as np




class Classifier_FCN(BaseNet):

    @BaseNet.layer
    def conv1d(self, name, x, num_filters, filter_size, stride, padding='same', relu_alpha=0.1):
        with tf.variable_scope(name) as scope:
            x = tf.layers.conv1d(x, num_filters, filter_size, stride, padding)
            x = tf.layers.batch_normalization(x, training=self.training)
            return tf.maximum(x, relu_alpha * x)  # leaky relu

    @BaseNet.layer
    def global_avg_pooling(self,name,x):
        with tf.variable_scope(name) as scope:
            shape = self.output.shape
            shape_x = int(shape[1])
            return tf.layers.average_pooling1d(x,shape_x,shape_x,padding='valid')

    @BaseNet.layer
    def fc(self, name, x, num_outputs, batch_norm=True, relu=True):
        with tf.variable_scope(name) as scope:
            x = tf.layers.dense(tf.reshape(x,
                                           [-1, reduce(lambda a, b: a * b, x.shape.as_list()[1:])]), num_outputs)
            if batch_norm: x = tf.layers.batch_normalization(x, training=self.training)
            if relu:
                x = tf.nn.relu(x)
            return x

    @BaseNet.layer
    def softmax(self, name, x):
        return tf.nn.softmax(x)

    def __init__(self,input_shape, nb_classes, verbose=False,fcn_name='fcns'):
        self.training = tf.placeholder_with_default(False, shape=None)

        #FCNs模型构建############
        super(Classifier_FCN, self).__init__(fcn_name,
                                     tf.placeholder(tf.float32, [None, 40,128],fcn_name+'/input'))
        self.conv1d('conv1',num_filters=128,filter_size=8,stride=1)
        self.conv1d('conv2',num_filters=256,filter_size=5,stride=1)
        self.conv1d('conv3',num_filters=128,filter_size=3,stride=1)
        self.global_avg_pooling('gap')
        self.fc('fc',nb_classes,batch_norm=False,relu=False)
        self.softmax('softmax')
        #########################

if __name__ == '__main__':
    FCN = Classifier_FCN([None,40,128],2)
    print('\nTotal number of parameters: {}'.format(FCN.total_params))
    print(FCN)