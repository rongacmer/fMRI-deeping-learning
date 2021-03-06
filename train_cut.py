import math, datetime, os
from voxnet import *
from fmri_data import fMRI_data
from config import cfg
from evaluation import evaluation
import nibabel as nib
import time
def main(cross_epoch = 0,data_index=None,brain_map=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data', num_batches = 512*5+1,test_size=6):
    tf.reset_default_graph()

    if cut_shape == None:
        brain_map = [212,213,214,215,216,217,218]
        cut_shape = [100,0,100,0,100,0]
        mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
        mask = mask.get_fdata()
        #获取截取的sMRI大小
        for x in brain_map:
            tmp = np.where(mask==x)
            for i in range(3):
                cut_shape[2*i] = min(cut_shape[2*i],np.min(tmp[i]))
                cut_shape[2 * i + 1] = max(cut_shape[2 * i+1], np.max(tmp[i]))
        print(cut_shape)

    xyz = 32
    dataset = fMRI_data(data_type, data_index=data_index,dir=pre_dir, batch_mode='random', varbass=cfg.varbass)
    voxnet = VoxNet(input_shape=[None,xyz,xyz,xyz,1],voxnet_type='cut')

    #数据权值
    data_value=[[1],[1]]
    #创建数据
    p = dict()  # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 2])
    p['data_value'] = tf.placeholder(tf.float32,[2,1])

    p['Weight'] = tf.matmul(p['labels'], p['data_value'])
    p['cross_loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=voxnet[-2], labels=p['labels'])
    p['Weight'] = tf.reshape(p['Weight'], [-1])
    p['x_loss'] = tf.multiply(p['Weight'], p['cross_loss'])
    p['loss'] = tf.reduce_mean(p['x_loss'])
    p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in voxnet.kernels])
    p['loss'] = p['loss'] + 0.0001 * p['l2_loss']

    p['prediction'] = tf.argmax(voxnet[-1], 1)
    p['y_true'] = tf.argmax(p['labels'], 1)
    p['correct_prediction'] = tf.equal(tf.argmax(voxnet[-1], 1), tf.argmax(p['labels'], 1))
    p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

    p['learning_rate'] = tf.placeholder(tf.float32)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        p['train'] = tf.train.AdamOptimizer(p['learning_rate'], epsilon=1e-3).minimize(p['loss'])
    p['weights_decay'] = tf.train.GradientDescentOptimizer(p['learning_rate']).minimize(p['l2_loss'])

    # Hyperparameters
    batch_size = 50

    initial_learning_rate = 0.0001
    min_learning_rate = 0.00001
    learning_rate_decay_limit = 0.0001

    num_batches_per_epoch = len(dataset.train) / batch_size
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch

    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308

    wait = 0
    paitence = 10
    accuracy_filename = os.path.join(cfg.voxnet_checkpoint_dir,'accuracies.txt')
    if not os.path.isdir(cfg.voxnet_checkpoint_dir):
        os.mkdir(cfg.voxnet_checkpoint_dir)

    with open(accuracy_filename, 'a') as f:
        f.write(str(brain_map)+'\n')

    filename = ""
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if cfg.istraining:
            voxnet.npz_saver.restore(session,cfg.voxnet_checkpoint)
        for batch_index in range(num_batches):

            # learning_rate = max(min_learning_rate,
            #                     initial_learning_rate * 0.8 ** (learning_step / learning_decay))
            learning_rate = 0.0001
            learning_step += 1

            # if batch_index > weights_decay_after and batch_index % 256 == 0:
            #     session.run(p['weights_decay'], feed_dict=feed_dict)

            voxs, labels = dataset.train.oversampling.get_smri_batch(cut_shape,batch_size)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels,voxnet.keep_prob:0.3,
                         p['learning_rate']: learning_rate, voxnet.training: True,p['data_value']:data_value,}

            session.run(p['train'], feed_dict=feed_dict)

            if batch_index and batch_index % 512 == 0:

                print("{} batch: {}".format(datetime.datetime.now(), batch_index))
                print('learning rate: {}'.format(learning_rate))

                feed_dict[voxnet.training] = False
                loss = session.run(p['loss'], feed_dict=feed_dict)
                print('loss: {}'.format(loss))

                if (batch_index and loss > 1.5 * min_loss and
                        learning_rate > learning_rate_decay_limit):
                    min_loss = loss
                    learning_step *= 1.2
                    print("decreasing learning rate...")
                min_loss = min(loss, min_loss)

            if batch_index and batch_index % 128 == 0:
                num_accuracy_batches = 30
                # train_accuracy = 0
                training_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.train.random_sampling.get_smri_batch(cut_shape,batch_size)
                    feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False,voxnet.keep_prob:1.0}
                    start_time = time.time()
                    predictions, y_true = session.run([p['prediction'], p['y_true']], feed_dict=feed_dict)
                    training_evaluation += evaluation(y_true=y_true,y_predict=predictions)
                    end_time =time.time()
                    print('total time: %f'%((end_time - start_time) / 60))
                    print(training_evaluation)
                # training_accuracy = total_accuracy / num_accuracy_batgetches
                print('training accuracy \n' + str(training_evaluation))
                num_accuracy_batches = test_size
                test_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.test.random_sampling.get_smri_batch(cut_shape,batch_size)
                    feed_dict = {voxnet[0]: voxs,  p['labels']: labels, voxnet.training: False,voxnet.keep_prob:1.0}
                    predictions, y_true = session.run([p['prediction'], p['y_true']], feed_dict=feed_dict)
                    test_evaluation += evaluation(y_true=y_true, y_predict=predictions)
                    # print(y_true, predictions)
                    print(test_evaluation)
                print('test accuracy \n' + str(test_evaluation))
                with open(accuracy_filename, 'a') as f:
                    f.write(str(checkpoint_num) + ':\n')
                    f.write(str(training_evaluation) + '\n')
                    f.write(str(test_evaluation) + '\n')

                if training_evaluation.ACC >= 0.95:
                    wait += 1
                    if wait >= paitence:
                        filename = 'cx-{}.npz'.format(cross_epoch)
                        filename = os.path.join(cfg.voxnet_checkpoint_dir, filename)
                        voxnet.npz_saver.save(session, filename)
                        print('checkpoint saved!')
                        return filename
                else:
                    wait = 0
                # if batch_index % 1024 == 0 or training_evaluation.ACC >= 0.9:
                #     print('saving checkpoint {}...'.format(checkpoint_num))
                #     filename = 'cx-{}.npz'.format(checkpoint_num)
                #     filename = os.path.join(cfg.voxnet_checkpoint_dir,filename)
                #     voxnet.npz_saver.save(session, filename)
                #     print('checkpoint saved!')
                #     checkpoint_num += 1
                #     if training_evaluation.ACC >= 0.9:
                #         break
        filename = 'cx-{}.npz'.format(cross_epoch)
        filename = os.path.join(cfg.voxnet_checkpoint_dir, filename)
        voxnet.npz_saver.save(session, filename)
        print('checkpoint saved!')
    return filename


if __name__ == '__main__':
    tf.app.run()
