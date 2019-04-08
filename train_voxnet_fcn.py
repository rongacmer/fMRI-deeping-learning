import math, datetime, os
from FCN import *
from voxnet import VoxNet
from fmri_data import fMRI_data
from config import cfg
import time
from evaluation import *
import nibabel as nib
import toolbox
import train_fcn
import train_cut

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def syn_train(*argv):

    data_value = [[1.0],[1.0]]
    time_dim = 80  # 挑选时间片个数
    batch_size = 8

    brain_map = [217]
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
    if not os.path.isdir(cfg.checkpoint_dir):
        os.mkdir(cfg.checkpoint_dir)

    if not os.path.exists(accuracy_filename):
        with open(accuracy_filename, 'a') as f:
            f.write('')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # voxnet.npz_saver.restore(session, cfg.voxnet_checkpoint_dir)
        #voxnet赋值
        for batch_index in range(num_batches):
            start = time.time()
            learning_rate = max(min_learning_rate,
                                initial_learning_rate * 0.8 ** (learning_step / learning_decay))
            learning_step += 1

            if batch_index > weights_decay_after and batch_index % 256 == 0:
                session.run(p['weights_decay'], feed_dict=feed_dict)

            voxs, labels = dataset.train.oversampling.get_fmri_brain(cut_shape,batch_size,time_dim)
            # voxs = np.zeros(input_shape)
            # labels = np.zeros((8,2))
            feed_dict = {voxnet[0]:voxs, p['labels']: labels,
                         p['learning_rate']: learning_rate,
                         FCNs.training: True,
                         voxnet.training:True,
                         p['data_value']:data_value}
            # print(voxs.shape,labels.shape)
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
                    voxs, labels = dataset.train.random_sampling.get_fmri_brain(cut_shape, batch_size, time_dim)
                    feed_dict = {voxnet[0]:voxs,p['labels']: labels, FCNs.training: False,voxnet.training:False}
                    total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
                training_accuracy = total_accuracy / num_accuracy_batches
                print('training accuracy: {}'.format(training_accuracy))
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
                with open(accuracy_filename, 'a') as f:
                    f.write(' '.join(map(str, (checkpoint_num, training_accuracy, total_evaluation))) + '\n')
                if batch_index % 256 == 0:
                    print('saving checkpoint {}...'.format(checkpoint_num))
                    filename = 'voxnet-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.checkpoint_dir, filename)
                    voxnet.npz_saver.save(session,filename)
                    filename = 'fcn-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.checkpoint_dir, filename)
                    FCNs.npz_saver.save(session, filename)
                    print('checkpoint saved!')
                    checkpoint_num += 1
            end = time.time()
            print('time:',(end-start)/60)

def train(_):

    brain_map = [217]
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
    #样本划分
    data_type=['MCIc','MCInc']
    test_len=[2,4]
    dir = '/home/anzeng/rhb/fmri_data'
    data_index={}
    iters={} #迭代器，用于留一测试
    epoch = 10
    #构造迭代器
    for cnt,i in enumerate(data_type):
        path = os.path.join(dir,i)
        data_len = len(os.listdir(path))
        iters[i]=iter(toolbox.cut_data(data_len,test_len[cnt]))

    accuracy = evaluation()
    f = open(cfg.output,'w')

    for step in range(epoch):
        # 数据迭代器
        print('################%d#################' % step)
        f.write('################{}#############'.format(step) + '\n')
        voxnet_data = {}
        fcn_data = {}
        for x in data_type:
            index = iters[x].__next__()
            print(index)
            f.write(str(index) + '\n')
            voxnet_data[x]={'train':index['voxnet_train'],'test':index['test']}
            fcn_data[x] = {'train':index['fcn_train'],'test':index['test']}
        #voxnent训练
        print('voxnet train start')
        voxnet_filename = train_cut.main(brain_map=brain_map,data_index=voxnet_data,cut_shape=cut_shape,pre_dir = dir,num_batches=2048* 5+1,test_size=6)
        print(voxnet_filename)
        print('voxnent train finish')
        #fcn训练
        print('fcn train start')
        accuracy += train_fcn.main(brain_map=brain_map,data_index=fcn_data,cut_shape=cut_shape,pre_dir=dir,num_batches=256*5+1,voxnet_point=voxnet_filename,test_size=6)
        print('fcn train finish')
        #输出测试准确率
        output = 'iter {}:{}'.format(step,accuracy)
        print(output)
        f.write(output+'\n')
    f.close()

def test(_):
    pass
if __name__ == '__main__':
    tf.app.run(train)
