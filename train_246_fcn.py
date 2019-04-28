import math, datetime, os
import nibabel as nib
from FCN import *
from fmri_data import fMRI_data
from config import cfg
from evaluation import *
import time

def main(data_index=None,brain_map=[218],data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data', num_batches = 512*5+1,test_size=6):
    tf.reset_default_graph()
    dataset = fMRI_data(data_index=data_index,data_type=data_type, varbass=False, dir=pre_dir)

    # 超参数设置

    num_batches = num_batches
    batch_size = 16

    initial_learning_rate = 0.0001
    min_learning_rate = 0.000001
    learning_rate_decay_limit = 0.0001

    num_batches_per_epoch = len(dataset.train) / float(batch_size)
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch

    feature_index = brain_map
    data_value=[[1.0],[1.0]]
    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308
    time_dim = 80
    varbass = True

    #模型框架
    mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
    mask = mask.get_fdata()
    f_len = 0
    for i in feature_index:
        f_len += len(np.where(mask == i)[0])
    FCNs = Classifier_FCN(tf.placeholder(tf.float32,[None,time_dim,25]),nb_classes=2)


    # 创建数据
    p = dict()  # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 2])
    p['data_value'] = tf.placeholder(tf.float32, [2, 1])

    p['Weight'] = tf.matmul(p['labels'], p['data_value'])
    p['cross_loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=FCNs[-2], labels=p['labels'])
    p['Weight'] = tf.reshape(p['Weight'],[-1])
    p['x_loss'] = tf.multiply(p['Weight'], p['cross_loss'])
    p['loss'] = tf.reduce_mean(p['x_loss'])
    p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in FCNs.kernels])

    p['prediction'] = tf.argmax(FCNs[-1],1)
    p['y_true'] = tf.argmax(p['labels'],1)
    p['correct_prediction'] = tf.equal(tf.argmax(FCNs[-1], 1), tf.argmax(p['labels'], 1))
    p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

    p['learning_rate'] = tf.placeholder(tf.float32)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        p['train'] = tf.train.AdamOptimizer(0.00001, epsilon=1e-3).minimize(p['loss'])
    p['weights_decay'] = tf.train.GradientDescentOptimizer(0.0001).minimize(p['l2_loss'])

    accuracy_filename = os.path.join(cfg.fcn_checkpoint_dir, 'accuracies.txt')
    if not os.path.isdir(cfg.fcn_checkpoint_dir):
        os.mkdir(cfg.fcn_checkpoint_dir)

    with open(accuracy_filename, 'a') as f:
        f.write(str(feature_index))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # voxnet_data = np.ones([1,61,73,61,1],np.float32)
        for batch_index in range(num_batches):

            # learning_rate = max(min_learning_rate,
            #                     initial_learning_rate * 0.5 ** (learning_step / learning_decay))
            learning_step += 1

            if batch_index > weights_decay_after and batch_index % 256 == 0:
                session.run(p['weights_decay'], feed_dict=feed_dict)

            voxs, labels = dataset.train.oversampling.get_brain_batch(mask,batch_size=batch_size,time_dim=time_dim,feature_index=feature_index)
            feed_dict = {FCNs[0]: voxs, p['labels']: labels,FCNs.training: True,p['data_value']:data_value}

            Weight,cross_loss,x_loss,_ = session.run([p['Weight'],p['cross_loss'],p['x_loss'],p['train']], feed_dict=feed_dict)
            # print("Weight\n",Weight)
            # print("cross_loss\n",cross_loss)
            # print("x_loss\n",x_loss)
            if batch_index and batch_index % 64 == 0:

                print("{} batch: {}".format(datetime.datetime.now(), batch_index))
                # print('learning rate: {}'.format(learning_rate))

                feed_dict[FCNs.training] = False
                loss = session.run(p['loss'], feed_dict=feed_dict)
                print('loss: {}'.format(loss))

                # if (batch_index and loss > 1.5 * min_loss and
                #         learning_rate > learning_rate_decay_limit):
                #     min_loss = loss
                #     learning_step *= 1.2
                #     print("decreasing learning rate...")
                # min_loss = min(loss, min_loss)

            if batch_index and batch_index % 64 == 0:
                num_accuracy_batches = 50
                train_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.train.random_sampling.get_brain_batch(mask, batch_size=batch_size, time_dim=time_dim,feature_index=feature_index)
                    feed_dict = {FCNs[0]: voxs,p['labels']: labels, FCNs.training: False}
                    start_time = time.time()
                    predictions, y_true = session.run([p['prediction'], p['y_true']], feed_dict=feed_dict)
                    train_evaluation += evaluation(y_true=y_true, y_predict=predictions)
                    end_time = time.time()
                    print('total time: %f' % ((end_time - start_time) / 60))
                    print(train_evaluation)
                print('training accuracy \n' + str(train_evaluation))
                # num_accuracy_batches = 10
                print('loss: {}'.format(loss))
                num_accuracy_batches = test_size
                test_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.test.random_sampling.get_brain_batch(mask, batch_size=1, time_dim=time_dim,feature_index=feature_index)
                    feed_dict = {FCNs[0]: voxs,p['labels']: labels, FCNs.training: False}
                    y_true,prediction = session.run([p['y_true'],p['prediction']], feed_dict=feed_dict)
                    test_evaluation += evaluation(y_true=y_true,y_predict=prediction)
                    # print(y_true, prediction)
                    print(test_evaluation)
                print('test accuracy \n' + str(test_evaluation))
                with open(accuracy_filename, 'a') as f:
                    f.write(str(checkpoint_num) + ':\n')
                    f.write(str(train_evaluation) + '\n')
                    f.write(str(test_evaluation) + '\n')
                # fr.write('test accuracy: {}'.format(test_accuracy))
                if batch_index % 1024 == 0:
                    print('saving checkpoint {}...'.format(checkpoint_num))
                    filename = 'cx-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.fcn_checkpoint_dir, filename)
                    FCNs.npz_saver.save(session, filename)
                    print('checkpoint saved!')
                    checkpoint_num += 1
                if train_evaluation.ACC > 0.95 and batch_index % 1024 == 0:
                    break
    return test_evaluation


if __name__ == '__main__':
    tf.app.run()
