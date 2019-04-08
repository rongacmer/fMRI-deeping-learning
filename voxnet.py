from basenet import *


class VoxNet(BaseNet):

    @BaseNet.layer
    def conv3d(self, name, x, num_filters, filter_size, stride, padding='valid', relu_alpha=0.1):
        with tf.variable_scope(name) as scope:
            x = tf.layers.conv3d(x, num_filters, filter_size, stride, padding)
            x = tf.layers.batch_normalization(x, training=self.training)
            return tf.maximum(x, relu_alpha * x)  # leaky relu

    @BaseNet.layer
    def max_pool3d(self, name, x, size=2, stride=2, padding='valid'):
        return tf.layers.max_pooling3d(x, size, stride, padding)

    @BaseNet.layer
    def avg_pool3d(self, name, x, size=2, stride=2, padding='valid'):
        return tf.layers.average_pooling3d(x, size, stride, padding)


    @BaseNet.layer
    def fc(self, name, x, num_outputs, batch_norm=True, relu=True):
        with tf.variable_scope(name) as scope:
            x = tf.layers.dense(tf.reshape(x,
                                           [-1, reduce(lambda a, b: a * b, x.shape.as_list()[1:])]), num_outputs)
            if batch_norm: x = tf.layers.batch_normalization(x, training=self.training)
            if relu: x = tf.nn.relu(x)
            return x

    @BaseNet.layer
    def softmax(self, name, x):
        return tf.nn.softmax(x)

    def __init__(self,input_shape, voxnet_name='voxnet',voxnet_type='all_conv'):
        self.training = tf.placeholder_with_default(False, shape=None)
        super(VoxNet, self).__init__(voxnet_name,
                                     tf.placeholder(tf.float32,input_shape,name=voxnet_name+'/input'))

        if voxnet_type == 'cut':
            # self.conv3d('conv1', 32, 5, 2)
            # self.conv3d('conv2', 32, 3, 1)
            # self.max_pool3d('max_pool')
            # self.fc('fc1', 128)
            # self.fc('fc2', 40, batch_norm=False, relu=False)
            # self.softmax('softmax')
            # output = self.output.shape
            self.conv3d('conv1', 64, 5, 2,padding = 'same')
            self.conv3d('conv2', 64, 3, 1,padding = 'same')
            self.conv3d('conv3', 128, 2, 2,padding = 'same')
            self.conv3d('conv4', 128, 2, 2,padding = 'same')
            # 全局池大小
            shape = self.output.shape
            shape_x = [shape[i] for i in range(1, 4)]
            self.avg_pool3d('gap', size=shape_x, stride=shape_x)
            # self.fc('fc1', 128)
            self.fc('fc1', 40, batch_norm=False, relu=False)
            self.fc('fc2', 2, batch_norm=False, relu=False)
            self.softmax('softmax')

        elif voxnet_type == 'all_conv':
            # output = self.output.shape
            self.conv3d('conv1', 64, 5, 2)
            # output = self.output.shape
            self.conv3d('conv2', 64, 3, 1)
            # output = self.output.shape
            self.conv3d('conv3', 128, 2, 2)
            # output = self.output.shape
            self.conv3d('conv4', 128, 2, 2)
            #全局池大小
            shape = self.output.shape
            shape_x = [shape[i] for i in range(1,4)]
            self.avg_pool3d('gap',size=shape_x,stride=shape_x)
            # self.fc('fc1', 128)
            self.fc('fc1',40, batch_norm=False, relu=False)
            self.fc('fc3', 2, batch_norm=False, relu=False)
            self.softmax('softmax')


if __name__ == '__main__':
    voxnet = VoxNet(voxnet_name='vx',input_shape=[None,23,23,23,1])
    print(voxnet)
    # print('\nTotal number of parameters: {}'.format(voxnet.total_params))