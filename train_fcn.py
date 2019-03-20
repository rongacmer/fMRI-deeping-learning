import math, datetime, os
from FCN import *
from voxnet import VoxNet
from fmri_data import fMRI_data
from config import cfg
import time


def main(*argv):
    # fr = open(cfg.output, 'w')
    dataset = fMRI_data(['AD', 'NC'],varbass=False,dir="/home/anzeng/rhb/fmri_data")
    voxnet = VoxNet()
    FCNs = Classifier_FCN(input_shape=[None,40,128],nb_classes=2)

    # 创建数据
    p = dict()  # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 2])

    p['loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=FCNs[-2], labels=p['labels'])
    p['loss'] = tf.reduce_mean(p['loss'])
    p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in FCNs.kernels])

    p['correct_prediction'] = tf.equal(tf.argmax(FCNs[-1], 1), tf.argmax(p['labels'], 1))
    p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

    p['learning_rate'] = tf.placeholder(tf.float32)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        p['train'] = tf.train.AdamOptimizer(p['learning_rate'], epsilon=1e-3).minimize(p['loss'])
    p['weights_decay'] = tf.train.GradientDescentOptimizer(p['learning_rate']).minimize(p['l2_loss'])

    # p['test_error'] = tf.placeholder(tf.float32)
    # 超参数设置

    num_batches = 2147483647
    batch_size = 8

    initial_learning_rate = 0.001
    min_learning_rate = 0.000001
    learning_rate_decay_limit = 0.0001

    num_batches_per_epoch = len(dataset.train) / float(batch_size)
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch

    time_dim = 40 #挑选时间片个数
    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308

    accuracy_filename = os.path.join(cfg.checkpoint_dir, 'accuracies.txt')
    if not os.path.isdir(cfg.checkpoint_dir):
        os.mkdir(cfg.checkpoint_dir)

    if not os.path.exists(accuracy_filename):
        with open(accuracy_filename, 'w') as f:
            f.write('')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # voxnet.npz_saver.restore(session, cfg.voxnet_checkpoint_dir)
        #voxnet赋值
        voxnet_data = np.ones([1,61,73,61,1],np.float32)
        for batch_index in range(num_batches):
            start = time.time()
            learning_rate = max(min_learning_rate,
                                initial_learning_rate * 0.5 ** (learning_step / learning_decay))
            learning_step += 1

            if batch_index > weights_decay_after and batch_index % 256 == 0:
                session.run(p['weights_decay'], feed_dict=feed_dict)

            voxs, labels = dataset.train.get_time_batch(session,voxnet,time_dim=time_dim,batch_size=batch_size)
            feed_dict = {FCNs[0]: voxs,voxnet[0]:voxnet_data, p['labels']: labels,
                         p['learning_rate']: learning_rate, FCNs.training: True}

            session.run(p['train'], feed_dict=feed_dict)

            if batch_index and batch_index % 32 == 0:

                print("{} batch: {}".format(datetime.datetime.now(), batch_index))
                print('learning rate: {}'.format(learning_rate))
                # fr.write("{} batch: {}".format(datetime.datetime.now(), batch_index))
                # fr.write('learning rate: {}'.format(learning_rate))

                feed_dict[FCNs.training] = False
                loss = session.run(p['loss'], feed_dict=feed_dict)
                print('loss: {}'.format(loss))

                if (batch_index and loss > 1.5 * min_loss and
                        learning_rate > learning_rate_decay_limit):
                    min_loss = loss
                    learning_step *= 1.2
                    print("decreasing learning rate...")
                min_loss = min(loss, min_loss)

            if batch_index and batch_index % 16 == 0:
                num_accuracy_batches = 10
                total_accuracy = 0
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.train.get_time_batch(session,voxnet,time_dim=time_dim,batch_size=batch_size)
                    feed_dict = {FCNs[0]: voxs, voxnet[0]:voxnet_data,p['labels']: labels, FCNs.training: False}
                    total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
                training_accuracy = total_accuracy / num_accuracy_batches
                print('training accuracy: {}'.format(training_accuracy))
                num_accuracy_batches = 10
                total_accuracy = 0
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.test.get_time_batch(session,voxnet,time_dim=time_dim,batch_size=batch_size)
                    feed_dict = {FCNs[0]: voxs,voxnet[0]:voxnet_data, p['labels']: labels, FCNs.training: False}
                    total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
                test_accuracy = total_accuracy / num_accuracy_batches
                print('test accuracy: {}'.format(test_accuracy))
                # fr.write('test accuracy: {}'.format(test_accuracy))
                with open(accuracy_filename, 'a') as f:
                    f.write(' '.join(map(str, (checkpoint_num, training_accuracy, test_accuracy))) + '\n')
                if batch_index % 256 == 0:
                    print('saving checkpoint {}...'.format(checkpoint_num))
                    filename = 'cx-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.checkpoint_dir, filename)
                    FCNs.npz_saver.save(session, filename)
                    print('checkpoint saved!')
                    checkpoint_num += 1
            end = time.time()
            print(time.time(),(end-start)/60)


if __name__ == '__main__':
    tf.app.run()
