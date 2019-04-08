import math, datetime, os
from FCN import *
from voxnet import VoxNet
from fmri_data import fMRI_data
from config import cfg
import time
from evaluation import *


def main(data_index=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data',
         num_batches = 256*5,voxnet_point=None,test_size = 6,brain_map=[217]):
    # fr = open(cfg.output, 'w')
    tf.reset_default_graph()

    time_dim = 80  # 挑选时间片个数
    batch_size = 8

    dataset = fMRI_data(data_type,data_index=data_index,varbass=False,dir=pre_dir)
    input_shape = [None]
    for i in range(3):
        input_shape.append(cut_shape[2 * i + 1] + 1 - cut_shape[2 * i])
    input_shape.append(1)
    print(input_shape)
    voxnet = VoxNet(input_shape=input_shape, voxnet_type='cut')
    FCNs = Classifier_FCN(tf.placeholder(tf.float32,[None,time_dim,128]),nb_classes=2)

    data_value = [[1], [1]]

    # 创建数据
    p = dict()  # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 2])
    p['data_value'] = tf.placeholder(tf.float32, [2, 1])

    p['Weight'] = tf.matmul(p['labels'], p['data_value'])
    p['cross_loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=FCNs[-2], labels=p['labels'])
    p['Weight'] = tf.reshape(p['Weight'], [-1])
    p['x_loss'] = tf.multiply(p['Weight'], p['cross_loss'])
    p['loss'] = tf.reduce_mean(p['x_loss'])
    p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in FCNs.kernels])

    p['prediction'] = tf.argmax(FCNs[-1],1)
    p['y_true'] = tf.argmax(p['labels'],1)
    p['correct_prediction'] = tf.equal(p['prediction'], p['y_true'])
    p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

    p['learning_rate'] = tf.placeholder(tf.float32)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        p['train'] = tf.train.AdamOptimizer(p['learning_rate'], epsilon=1e-3).minimize(p['loss'])
    p['weights_decay'] = tf.train.GradientDescentOptimizer(p['learning_rate']).minimize(p['l2_loss'])

    # p['test_error'] = tf.placeholder(tf.float32)
    # 超参数设置


    initial_learning_rate = 0.01
    min_learning_rate = 0.0001
    learning_rate_decay_limit = 0.0001

    num_batches_per_epoch = len(dataset.train) / float(batch_size)
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch


    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308

    if voxnet_point:
        cfg.voxnet_checkpoint = voxnet_point

    accuracy_filename = os.path.join(cfg.fcn_checkpoint_dir, 'accuracies.txt')
    if not os.path.isdir(cfg.fcn_checkpoint_dir):
        os.mkdir(cfg.fcn_checkpoint_dir)

    if not os.path.exists(accuracy_filename):
        with open(accuracy_filename, 'a') as f:
            f.write('')
    with open(accuracy_filename,'a') as f:
        f.write(str(brain_map)+'\n')

    #返回值
    total_evaluation = evaluation()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        voxnet.npz_saver.restore(session, cfg.voxnet_checkpoint)
        #voxnet赋值
        input_shape[0]=1
        voxnet_data = np.ones(input_shape,np.float32)
        input_shape[0]=-1
        for batch_index in range(num_batches):
            start = time.time()
            learning_rate = max(min_learning_rate,
                                initial_learning_rate * 0.5 ** (learning_step / learning_decay))
            learning_step += 1

            if batch_index > weights_decay_after and batch_index % 256 == 0:
                session.run(p['weights_decay'], feed_dict=feed_dict)

            voxs, labels = dataset.train.oversampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size=batch_size)
            feed_dict = {FCNs[0]: voxs,voxnet[0]:voxnet_data, p['labels']: labels,
                         p['learning_rate']: learning_rate, FCNs.training: True,p['data_value']:data_value}

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
                    voxs, labels = dataset.train.random_sampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size=batch_size)
                    feed_dict = {FCNs[0]: voxs, voxnet[0]:voxnet_data,p['labels']: labels, FCNs.training: False}
                    total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
                training_accuracy = total_accuracy / num_accuracy_batches
                print('training accuracy: {}'.format(training_accuracy))
                num_accuracy_batches = test_size
                total_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.test.random_sampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size=batch_size)
                    feed_dict = {FCNs[0]: voxs,voxnet[0]:voxnet_data, p['labels']: labels, FCNs.training: False}
                    predictions,y_true = session.run([p['prediction'],p['y_true']], feed_dict=feed_dict)
                    total_evaluation += evaluation(y_true=y_true,y_predict=predictions)
                    print(total_evaluation)
                print('test accuracy \n'+str(total_evaluation))
                # fr.write('test accuracy: {}'.format(test_accuracy))
                with open(accuracy_filename, 'a') as f:
                    f.write(' '.join(map(str, (checkpoint_num, training_accuracy, total_evaluation))) + '\n')
                if batch_index % 256 == 0:
                    print('saving checkpoint {}...'.format(checkpoint_num))
                    filename = 'cx-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.fcn_checkpoint_dir, filename)
                    FCNs.npz_saver.save(session, filename)
                    print('checkpoint saved!')
                    checkpoint_num += 1
            end = time.time()
            print('time:',(end-start)/60)
    return total_evaluation


if __name__ == '__main__':
    tf.app.run()
