import math, datetime, os
import nibabel as nib
from FCN import *
from fmri_data import fMRI_data
from config import cfg
from evaluation import *

def main(*argv):
    dataset = fMRI_data(['MCIc', 'MCInc'], data_value=[0.7,0.3],varbass=False, dir="/home/anzeng/rhb/fmri_data")

    # 超参数设置

    num_batches = 2147483647
    batch_size = 16

    initial_learning_rate = 0.001
    min_learning_rate = 0.000001
    learning_rate_decay_limit = 0.0001

    num_batches_per_epoch = len(dataset.train) / float(batch_size)
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch

    time_dim = 40  # 挑选时间片个数
    feature_index = [218]
    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308

    #模型框架
    mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
    mask = mask.get_fdata()
    f_len = 0
    for i in feature_index:
        f_len += len(np.where(mask == i)[0])
    FCNs = Classifier_FCN(input_shape=[None,40,f_len],nb_classes=2)

    # 创建数据
    p = dict()  # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 2])

    p['loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=FCNs[-2], labels=p['labels'])
    p['loss'] = tf.reduce_mean(p['loss'])
    p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in FCNs.kernels])

    p['prediction'] = tf.argmax(FCNs[-1],1)
    p['y_true'] = tf.argmax(p['labels'],1)
    p['correct_prediction'] = tf.equal(tf.argmax(FCNs[-1], 1), tf.argmax(p['labels'], 1))
    p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

    p['learning_rate'] = tf.placeholder(tf.float32)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        p['train'] = tf.train.AdamOptimizer(p['learning_rate'], epsilon=1e-3).minimize(p['loss'])
    p['weights_decay'] = tf.train.GradientDescentOptimizer(p['learning_rate']).minimize(p['l2_loss'])


    accuracy_filename = os.path.join(cfg.checkpoint_dir, 'accuracies.txt')
    if not os.path.isdir(cfg.checkpoint_dir):
        os.mkdir(cfg.checkpoint_dir)

    with open(accuracy_filename, 'a') as f:
        f.write(str(feature_index))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        voxnet_data = np.ones([1,61,73,61,1],np.float32)
        for batch_index in range(num_batches):

            learning_rate = max(min_learning_rate,
                                initial_learning_rate * 0.5 ** (learning_step / learning_decay))
            learning_step += 1

            if batch_index > weights_decay_after and batch_index % 256 == 0:
                session.run(p['weights_decay'], feed_dict=feed_dict)

            voxs, labels = dataset.train.get_brain_batch(mask,batch_size=batch_size,time_dim=time_dim,feature_index=feature_index)
            feed_dict = {FCNs[0]: voxs, p['labels']: labels,
                         p['learning_rate']: learning_rate, FCNs.training: True}

            session.run(p['train'], feed_dict=feed_dict)

            if batch_index and batch_index % 64 == 0:

                print("{} batch: {}".format(datetime.datetime.now(), batch_index))
                print('learning rate: {}'.format(learning_rate))

                feed_dict[FCNs.training] = False
                loss = session.run(p['loss'], feed_dict=feed_dict)
                print('loss: {}'.format(loss))

                if (batch_index and loss > 1.5 * min_loss and
                        learning_rate > learning_rate_decay_limit):
                    min_loss = loss
                    learning_step *= 1.2
                    print("decreasing learning rate...")
                min_loss = min(loss, min_loss)

            if batch_index and batch_index % 64 == 0:
                num_accuracy_batches = 10
                total_accuracy = 0
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.get_brain_batch(mask, batch_size=batch_size, time_dim=time_dim,feature_index=feature_index)
                    feed_dict = {FCNs[0]: voxs,p['labels']: labels, FCNs.training: False}
                    total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
                training_accuracy = total_accuracy / num_accuracy_batches
                print('training accuracy: {}'.format(training_accuracy))
                num_accuracy_batches = 10
                total_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.test.get_brain_batch(mask, batch_size=batch_size, time_dim=time_dim,feature_index=feature_index)
                    feed_dict = {FCNs[0]: voxs,p['labels']: labels, FCNs.training: False}
                    y_true,prediction = session.run([p['y_true'],p['prediction']], feed_dict=feed_dict)
                    total_evaluation += evaluation(y_true=y_true,y_predict=prediction)
                    print(y_true, prediction)
                    print(total_evaluation)
                test_evaluation = total_evaluation / num_accuracy_batches
                print('test accuracy \n' + str(test_evaluation))
                # fr.write('test accuracy: {}'.format(test_accuracy))
                with open(accuracy_filename, 'a') as f:
                    f.write(' '.join(map(str, (checkpoint_num, training_accuracy, test_evaluation))) + '\n')
                if batch_index % 1024 == 0:
                    print('saving checkpoint {}...'.format(checkpoint_num))
                    filename = 'cx-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.checkpoint_dir, filename)
                    FCNs.npz_saver.save(session, filename)
                    print('checkpoint saved!')
                    checkpoint_num += 1


if __name__ == '__main__':
    tf.app.run()
