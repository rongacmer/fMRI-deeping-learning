import math, datetime, os
from FCN import *
from voxnet import VoxNet
from fmri_data import fMRI_data
from config import cfg
import time
from evaluation import *
from sklearn import svm


def main(data_index=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data',
         num_batches = 256*5,voxnet_point=None,test_size = 6,brain_map=[217]):
    # fr = open(cfg.output, 'w')
    tf.reset_default_graph()

    time_dim = 80  # 挑选时间片个数
    batch_size = 8

    dataset = fMRI_data(data_type,data_index=data_index,varbass=False,dir=pre_dir)
    #SVM index

    #########################
    svm_index = {}
    train_len = 0
    test_len = 0
    for d_type in data_type:
        t_dir = os.path.join(pre_dir,d_type)
        t_len = os.listdir(t_dir)
        t_len = len(t_len)
        train_index = list(range(t_len))
        test_index = data_index[d_type]['test']
        for x in test_index:
            train_index.remove(x)
        _index = {'train':train_index,'test':test_index}
        train_len += len(train_index)
        test_len += len(test_index)
        svm_index[d_type] = _index
    print(train_len)
    print(test_len)
    print(svm_index)
    svm_dataset = fMRI_data(data_type,data_index = svm_index,varbass=False,dir=pre_dir)
    ##########################
    xyz = 32
    input_shape = [None,xyz,xyz,xyz,1]
    # for i in range(3):
    #     input_shape.append(cut_shape[2 * i + 1] + 1 - cut_shape[2 * i])
    # input_shape.append(1)
    # print(input_shape)
    voxnet = VoxNet(input_shape=input_shape, voxnet_type='cut')
    FCNs = Classifier_FCN(tf.placeholder(tf.float32,[None,time_dim,50]),nb_classes=2)

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
    min_learning_rate = 0.000001
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
    test_evaluation = evaluation()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        voxnet.npz_saver.restore(session, cfg.voxnet_checkpoint)
        #voxnet赋值
        input_shape[0]=1
        voxnet_data = np.ones(input_shape,np.float32)
        input_shape[0]=-1
        for batch_index in range(num_batches):
            start = time.time()
            # learning_rate = max(min_learning_rate,
            #                     initial_learning_rate * 0.5 ** (learning_step / learning_decay))
            learning_rate = 0.0001
            learning_step += 1

            if batch_index > weights_decay_after and batch_index % 256 == 0:
                session.run(p['weights_decay'], feed_dict=feed_dict)

            voxs, labels = dataset.train.oversampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size=batch_size)
            feed_dict = {FCNs[0]: voxs,voxnet[0]:voxnet_data,voxnet.keep_prob:1.0,FCNs.keep_prob:0.7, p['labels']: labels,
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
                num_accuracy_batches = 20
                train_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.train.random_sampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size=batch_size)
                    feed_dict = {FCNs[0]: voxs, voxnet[0]:voxnet_data,voxnet.keep_prob:1.0,FCNs.keep_prob:1.0,p['labels']: labels, FCNs.training: False}
                    predictions, y_true = session.run([p['prediction'], p['y_true']], feed_dict=feed_dict)
                    train_evaluation += evaluation(y_true=y_true, y_predict=predictions)
                print('training accuracy \n' + str(train_evaluation))
                num_accuracy_batches = test_size
                test_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.test.random_sampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size=batch_size)
                    feed_dict = {FCNs[0]: voxs,voxnet[0]:voxnet_data,voxnet.keep_prob:1.0,FCNs.keep_prob:1.0, p['labels']: labels, FCNs.training: False}
                    predictions,y_true = session.run([p['prediction'],p['y_true']], feed_dict=feed_dict)
                    test_evaluation += evaluation(y_true=y_true, y_predict=predictions)
                    print(test_evaluation)
                print('test accuracy \n'+str(test_evaluation))
                with open(accuracy_filename, 'a') as f:
                    f.write('checkpoint_num:' + str(checkpoint_num) + ':\n')
                    f.write('train:\n' + str(train_evaluation) + '\n')
                    f.write('test:\n' + str(test_evaluation) + '\n')
                if batch_index % 64 or train_evaluation.ACC >= 0.8 == 0:
                    ######SVM分类器####################
                    svm_feature = np.zeros((train_len+test_len,128))
                    svm_label = np.zeros(train_len+test_len)
                    for x in range(train_len):
                        voxs, labels = svm_dataset.train.random_sampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size = 1)
                        feed_dict = {FCNs[0]: voxs, voxnet[0]: voxnet_data,voxnet.keep_prob:1.0,FCNs.keep_prob:1.0, p['labels']: labels, FCNs.training: False}
                        feature,y_true = session.run([FCNs['gap'],p['y_true']],feed_dict = feed_dict)
                        feature = np.reshape(feature,[1,128])
                        svm_feature[x] = feature
                        # print(svm_feature[x])
                        svm_label[x] = y_true
                    for x in range(test_len):
                        voxs, labels = svm_dataset.test.random_sampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size = 1)
                        feed_dict = {FCNs[0]: voxs, voxnet[0]: voxnet_data,voxnet.keep_prob:1.0,FCNs.keep_prob:1.0, p['labels']: labels, FCNs.training: False}
                        feature,y_true = session.run([FCNs['gap'],p['y_true']],feed_dict = feed_dict)
                        feature = np.reshape(feature,[1, 128])
                        svm_feature[train_len + x] = feature
                        svm_label[train_len + x] = y_true
                    # print(svm_feature[0:train_len])
                    # print(svm_label[0:train_len])
                    clf = svm.SVC(C=1.0,kernel='rbf',gamma='auto')
                    clf.fit(svm_feature[0:train_len],svm_label[0:train_len])
                    predictions = clf.predict(svm_feature)
                    svm_train_evaluation = evaluation(y_true=svm_label[:train_len],y_predict=predictions[:train_len])
                    svm_test_evaluation = evaluation(y_true=svm_label[train_len:],y_predict=predictions[train_len:])
                    print('svm_train:\n'+str(svm_train_evaluation))
                    print('svm_test:\n' + str(svm_test_evaluation))
                    with open(accuracy_filename,'a') as f:
                        f.write('svm_train:\n' + str(svm_train_evaluation) + '\n')
                        f.write('svm_test:\n' + str(svm_test_evaluation) + '\n')
                    #################################################

                # fr.write('test accuracy: {}'.format(test_accuracy))

                if batch_index % 128 == 0 or train_evaluation.ACC >= 0.85:
                    print('saving checkpoint {}...'.format(checkpoint_num))
                    filename = 'cx-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.fcn_checkpoint_dir, filename)
                    FCNs.npz_saver.save(session, filename)
                    print('checkpoint saved!')
                    checkpoint_num += 1
                    if train_evaluation.ACC >= 0.85:
                        break
            end = time.time()
            print('time:',(end-start)/60)
    return test_evaluation


if __name__ == '__main__':
    tf.app.run()
