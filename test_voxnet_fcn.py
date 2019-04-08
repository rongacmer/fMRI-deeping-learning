import math, datetime, os
from FCN import *
from voxnet import VoxNet
from fmri_data import fMRI_data
from config import cfg
import time
from evaluation import *
import nibabel as nib


def main(*argv):

    data_value = [[1.0],[1.0]]
    time_dim = 80  # 挑选时间片个数
    batch_size = 8

    brain_map = [219]
    cut_shape = [100, 0, 100, 0, 100, 0]
    mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
    mask = mask.get_fdata()
    # 获取截取的sMRI大小
    for x in brain_map:
        tmp = np.where(mask == x)
        for i in range(3):
            cut_shape[2 * i] = min(cut_shape[2 * i], np.min(tmp[i]))
            cut_shape[2 * i + 1] = max(cut_shape[2 * i + 1], np.max(tmp[i]))
    print(cut_shape)

    dataset = fMRI_data(['AD', 'NC'], dir='/home/anzeng/rhb/fmri_data', batch_mode='random', varbass=cfg.varbass)
    input_shape = [time_dim*batch_size]
    for i in range(3):
        input_shape.append(cut_shape[2 * i + 1] + 1 - cut_shape[2 * i])
    input_shape.append(1)
    print(input_shape)
    # input_shape=[40,2,2,2,1]
    voxnet = VoxNet(input_shape=input_shape, voxnet_type='cut')
    FCN_input = tf.reshape(voxnet['gap'],(-1,time_dim,128))
    print(FCN_input)
    FCNs = Classifier_FCN(FCN_input,nb_classes=2)
    # 创建数据
    p = dict()  # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 2])
    p['prediction'] = tf.argmax(FCNs[-1],1)
    p['y_true'] = tf.argmax(p['labels'],1)


    # p['test_error'] = tf.placeholder(tf.float32)
    # 超参数设置

    num_batches = 2147483647

    initial_learning_rate = 0.01
    min_learning_rate = 0.0001
    learning_rate_decay_limit = 0.0001

    num_batches_per_epoch = len(dataset.train) / float(batch_size)
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch

    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308

    accuracy_filename = os.path.join(cfg.checkpoint_dir, 'accuracies.txt')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        voxnet.npz_saver.restore(session,cfg.voxnet_checkpoint_dir)
        FCNs.npz_saver.restore(session,cfg.fcn_checkpoint_dir)
        # voxnet.npz_saver.restore(session, cfg.voxnet_checkpoint_dir)
        #voxnet赋值
        num_accuracy_batches = 90
        total_evaluation = evaluation()
        for x in range(num_accuracy_batches):
            voxs, labels = dataset.train.random_sampling.get_fmri_brain(cut_shape, batch_size, time_dim)
            feed_dict = {voxnet[0]:voxs,p['labels']: labels, FCNs.training: False,voxnet.training:False}
            predictions, y_true = session.run([p['prediction'], p['y_true']], feed_dict=feed_dict)
            total_evaluation += evaluation(y_true=y_true, y_predict=predictions)
            print(total_evaluation)
        print('train accuracy \n' + str(total_evaluation))
        num_accuracy_batches = 15
        total_evaluation = evaluation()
        for x in range(num_accuracy_batches):
            voxs, labels = dataset.test.random_sampling.get_fmri_brain(cut_shape,batch_size,time_dim)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels, FCNs.training: False, voxnet.training: False}
            predictions,y_true = session.run([p['prediction'],p['y_true']], feed_dict=feed_dict)
            total_evaluation += evaluation(y_true=y_true,y_predict=predictions)
            print(total_evaluation)
        print('test accuracy \n'+str(total_evaluation))
        # fr.write('test accuracy: {}'.format(test_accuracy))


if __name__ == '__main__':
    tf.app.run()
