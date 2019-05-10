import math, datetime, os
from FCN import *
from voxnet import VoxNet
from fmri_data import fMRI_data
from config import cfg
import time
from evaluation import *
import nibabel as nib
import toolbox
import train_246_fcn
import train_fcn
import train_cut
import shutil
import convert
import keras_model
import smri_ensemble
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def syn_train(*argv):

    input_weight = [2,1,1,1]
    data_value = [[1.0],[1.0]]
    time_dim = 80  # 挑选时间片个数
    batch_size = 8

    brain_map = [217,218,219]
    cut_shape = [100, 0, 100, 0, 100, 0]
    mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
    mask = mask.get_fdata()
    # 获取截取的sMRI大小
    for x in brain_map:
        tmp = np.where(mask == x)
        for i in range(3):
            cut_shape[2 * i] = min(cut_shape[2 * i], np.min(tmp[i]))
            cut_shape[2 * i + 1] = max(cut_shape[2 * i + 1], np.max(tmp[i]))
    print(brain_map,cut_shape)

    dataset = fMRI_data(['AD', 'NC'], dir='/home/anzeng/rhb/fmri_data', batch_mode='random', varbass=cfg.varbass)
    input_shape = [None,32,32,32,1]
    # for i in range(3):
    #     input_shape.append(cut_shape[2 * i + 1] + 1 - cut_shape[2 * i])
    # input_shape.append(1)
    print(input_shape)
    # input_shape=[40,2,2,2,1]
    voxnet = VoxNet(input_shape=input_shape, voxnet_type='all_conv')
    FCN_input = tf.reshape(voxnet['gap'],(-1,time_dim,128))
    print(FCN_input)
    FCNs = Classifier_FCN(FCN_input,nb_classes=2)
    # 创建数据
    p = dict()  # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 2])
    p['data_value'] = tf.placeholder(tf.float32, [2, 1])
    p['input_pw'] = tf.placeholder(tf.float32,[None]) #预测权值


    p['Weight'] = tf.matmul(p['labels'], p['data_value'])
    p['cross_loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=FCNs[-2], labels=p['labels'])
    p['Weight'] = tf.reshape(p['Weight'], [-1])
    p['x_loss'] = tf.multiply(p['Weight'], p['cross_loss'])
    p['loss'] = tf.reduce_mean(p['x_loss'])
    p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in FCNs.kernels])

    # 预测时进行集成
    p['p_w'] = tf.reshape(p['input_pw'], [-1, 1])
    p['sum_w'] = tf.reduce_sum(p['p_w'])
    p['test_prediction'] = tf.cast(tf.argmax(FCNs[-1],1),tf.float32)
    p['test_prediction'] = tf.reshape(p['test_prediction'], [-1, 4])
    p['test_prediction'] = tf.matmul(p['test_prediction'], p['p_w'])
    p['test_prediction'] = tf.round(tf.divide(p['test_prediction'], p['sum_w']))

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

    checkpoint_num = cfg.checkpoint_start_num
    learning_step = 0
    min_loss = 1e308

    accuracy_filename = os.path.join(cfg.fcn_checkpoint_dir, 'accuracies.txt')
    if not os.path.isdir(cfg.fcn_checkpoint_dir):
        os.mkdir(cfg.fcn_checkpoint_dir)

    if not os.path.exists(accuracy_filename):
        with open(accuracy_filename, 'a') as f:
            f.write('')
    with open(accuracy_filename,'a') as f:
        f.write(str(brain_map)+'\n')
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if cfg.istraining:
            voxnet.npz_saver.restore(session, cfg.voxnet_checkpoint)
            FCNs.npz_saver.restore(session,cfg.fcn_checkpoint)
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
                num_accuracy_batches = 20
                train_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.train.random_sampling.get_fmri_brain(cut_shape, 5, time_dim)
                    feed_dict = {voxnet[0]:voxs,p['labels']: labels, FCNs.training: False,voxnet.training:False,p['input_pw']:input_weight}
                    predictions, y_true = session.run([p['test_prediction'], p['y_true']], feed_dict=feed_dict)
                    train_evaluation += evaluation(y_true=y_true, y_predict=predictions)
                    print(train_evaluation)
                print('train accuracy \n'+str(train_evaluation))
                num_accuracy_batches = 27
                test_evaluation = evaluation()
                for x in range(num_accuracy_batches):
                    voxs, labels = dataset.test.random_sampling.get_fmri_brain(cut_shape,1,time_dim)
                    feed_dict = {voxnet[0]: voxs, p['labels']: labels, FCNs.training: False, voxnet.training: False,p['input_pw']:input_weight}
                    predictions,y_true = session.run([p['test_prediction'],p['y_true']], feed_dict=feed_dict)
                    test_evaluation += evaluation(y_true=y_true,y_predict=predictions)
                    print(test_evaluation)
                print('test accuracy \n'+str(test_evaluation))
                # fr.write('test accuracy: {}'.format(test_accuracy))
                with open(accuracy_filename, 'a') as f:
                    f.write(str(checkpoint_num) + ':\n')
                    f.write(str(train_evaluation) + '\n')
                    f.write(str(test_evaluation) + '\n')
                if batch_index % 256 == 0:
                    print('saving checkpoint {}...'.format(checkpoint_num))
                    filename = 'voxnet-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.fcn_checkpoint_dir, filename)
                    voxnet.npz_saver.save(session,filename)
                    filename = 'fcn-{}.npz'.format(checkpoint_num)
                    filename = os.path.join(cfg.fcn_checkpoint_dir, filename)
                    FCNs.npz_saver.save(session, filename)
                    print('checkpoint saved!')
                    checkpoint_num += 1
            end = time.time()
            print('time:',(end-start)/60)

def train(_):

    # brain_map = [117,109,110,215,216,217,218,211,212]
    brain_map = [215,216,217,218]
    cut_shape = [100, 0, 100, 0, 100, 0]
    mask = nib.load('BN_Atlas_246_3mm.nii')
    mask = mask.get_fdata()

    # 获取截取的sMRI大小
    for x in brain_map:
        tmp = np.where(mask == x)
        for i in range(3):
            cut_shape[2 * i] = min(cut_shape[2 * i], np.min(tmp[i]))
            cut_shape[2 * i + 1] = max(cut_shape[2 * i + 1], np.max(tmp[i]))
    print(cut_shape)



    #样本划分
    data_type=['AD','NC']
    test_len=[10,10]
    dir = '/home/anzeng/rhb/fmri_data/new_fmri'
    target_sMRI_dir = '/home/anzeng/rhb/generate_sMRI'
    iters={} #迭代器，用于留一测试
    sample={} #样本号
    epoch = 10

    for i in data_type:
        sample_dir = '/home/anzeng/rhb/fmri_data/{}_sample.txt'.format(i)
        f_sample = open(sample_dir, 'r')
        lists = f_sample.readline()
        lists = lists[1:-1]
        lists = lists.split(', ')
        lists = list(map(lambda x: x[1:13], lists))
        sample[i] = lists
        # for j in lists:
        #     if 'sub-OAS30086' in j or 'sub-OAS30436' in j:
        #         print(j)
    print(sample)
    #构造迭代器
    for cnt,i in enumerate(data_type):
        # path = os.path.join(dir,i)
        # data_len = len(os.listdir(path))
        data_len = len(sample[i])
        # print(data_len)
        iters[i]=iter(toolbox.cut_data(data_len,test_len[cnt]))

    BGRU_ACC = evaluation() #模型
    BGRU_SVM_ACC = evaluation() #SVM模型
    ensemble_ACC = evaluation() #集成模型
    ensemble_SVM_ACC= evaluation() #SVM集成模型
    f = open(cfg.output,'w')

    #训练一个自编码器
    print('start autoencoder')
    start_time = time.time()
    smri_data = {}
    for d_type in data_type:
        # raw_dir = os.path.join(dir,d_type)
        class_sMRI_dir = os.path.join(target_sMRI_dir, d_type)
        smri_data[d_type] = {'train': range(len(os.listdir(class_sMRI_dir))), 'test': range(len(os.listdir(class_sMRI_dir)))}
    voxnet_filename = keras_model.train_autoencoder(cross_epoch=0, brain_map=brain_map, cut_shape=cut_shape,
                                                    data_type=data_type, data_index=smri_data, pre_dir=target_sMRI_dir,
                                                    num_batches=2000,
                                                    test_size=10)
    end_time = time.time()
    print('finish autoencoder:{}'.format((end_time-start_time)/60))

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
            # fcn_data[x]['train'] = np.hstack((voxnet_data[x]['train'],fcn_data[x]['train']))
        # 把fmri分过为sMRI
        print('start separate')
        start_time = time.time()
        smri_data = {}
        p_list = {}
        for d_type in data_type:
            # raw_dir = os.path.join(dir,d_type)
            class_sMRI_dir = os.path.join(target_sMRI_dir,d_type)
            # if os.path.isdir(class_sMRI_dir):
            #     shutil.rmtree(class_sMRI_dir)
            # os.makedirs(class_sMRI_dir)
            # # train_list = voxnet_data[d_type]['train']
            # # test_list = np.hstack((voxnet_data[d_type]['test'] ,fcn_data[d_type]['train']))
            # # for remove in test_list:
            # #     if remove in train_list:
            # #         np.delete(test_list,remove)
            # # train_list = fcn_data[d_type]['train']
            # # test_list = fcn_data[d_type]['test']
            p_list['smri_train'] = []
            p_list['smri_test'] = []
            p_list['train'] = voxnet_data[d_type]['train']
            p_list['train'] = list(map(lambda x:sample[d_type][x],p_list['train']))
            p_list['test'] = list((set(fcn_data[d_type]['train']) - set(voxnet_data[d_type]['train']))| set(fcn_data[d_type]['test']))
            p_list['test'] = list(map(lambda x: sample[d_type][x], p_list['test']))
            #构造训练集、测试及集
            for cnt,per in enumerate(os.listdir(class_sMRI_dir)):
                if per[0:12] in p_list['train']:
                    p_list['smri_train'].append(cnt)
                else:
                    p_list['smri_test'].append(cnt)
            smri_data[d_type] = {'train':p_list['smri_train'],'test':p_list['smri_test']}
            # print(train_list)
            # print(test_list)
            # convert.covert2smri(raw_dir,class_sMRI_dir,train_list,test_list,brain_map,mask)
            class_sMRI_dir = os.path.join(dir,d_type)
            print(d_type+' convert success')
        end_time = time.time()
        print('separate success total time:%f'%((end_time-start_time)/60))

        #时间集合
        time_data = {}
        for i in data_type:
            train_name = list(map(lambda x:sample[i][x],fcn_data[i]['train']))
            # print(train_name)
            test_name = list(map(lambda x:sample[i][x],fcn_data[i]['test']))
            train_list = []
            test_list = []
            data_dir = os.path.join(dir,i)
            for cnt,l in enumerate(os.listdir(data_dir)):
                # print(cnt,l)
                if l[0:12] not in train_name and l[0:12] not in test_name:
                    print(l)
                if l[0:12] in train_name:
                    train_list.append(cnt)
                else:
                    test_list.append(cnt)
            time_data[i]={'train':train_list,'test':test_list}
        # print(time_data)
        #############
        # voxnent训练
        # print('voxnet train start')
        # voxnet_filename = train_cut.main(cross_epoch=step,brain_map=brain_map,cut_shape=cut_shape,data_type=data_type,pre_dir = target_sMRI_dir,num_batches= 1024 * 5 + 1,test_size=10)
        # voxnet_filename = keras_model.train_autoencoder(cross_epoch=step, brain_map=brain_map, cut_shape=cut_shape,
        #                                  data_type=data_type, data_index=smri_data,pre_dir=target_sMRI_dir, num_batches=1000,
        #                                  test_size=10)
        print(voxnet_filename)
        # print('voxnent train finish')
        # voxnet_filename = '/home/anzeng/rhb/fmri/ADvsNC_voxnent_checkpoint_4_15/cx-0.npz'
        # fcn训练
        print('fcn train start')
        # one_accuracy = smri_ensemble.emsemble(cross_epoch=step,brain_map=brain_map,data_type=data_type,data_index = fcn_data,cut_shape=cut_shape,pre_dir=dir,num_batches=128*5+1,voxnet_point=voxnet_filename,test_size=20,f_handle=f)
        BGRU_ACC_One,BGRU_SVM_ACC_One = keras_model.train_GRUs(cross_epoch=step,brain_map=brain_map,data_type=data_type,data_index = time_data,cut_shape=cut_shape,pre_dir=dir,num_batches=128*5,voxnet_point=voxnet_filename,test_size=20,f_handle=f)
        BGRU_ACC += BGRU_ACC_One
        # print('BGRU_SVM_ACC')
        BGRU_SVM_ACC += BGRU_SVM_ACC_One
        # print('ensemble_SVM_ACC')
        # ensemble_SVM_ACC += smri_ensemble.svm_emsemble(cross_epoch=step,brain_map=brain_map,data_type=data_type,data_index = fcn_data,cut_shape=cut_shape,pre_dir=dir,num_batches=128*5+1,voxnet_point=voxnet_filename,test_size=20,f_handle=f)
        # print('ensemble_ACC')
        # ensemble_ACC += smri_ensemble.emsemble(cross_epoch=step,brain_map=brain_map,data_type=data_type,data_index = fcn_data,cut_shape=cut_shape,pre_dir=dir,num_batches=128*5+1,voxnet_point=voxnet_filename,test_size=20,f_handle=f)
        # accuracy += train_fcn.main(brain_map=brain_map, data_type = data_type,data_index=fcn_data, cut_shape=cut_shape, pre_dir=dir,
        #                            num_batches=128 * 5 + 1, voxnet_point=voxnet_filename, test_size=20)
        # train_fcn.main(brain_map=brain_map,cut_shape=cut_shape,pre_dir=dir,data_type=data_type,num_batches=256*5+1,voxnet_point=voxnet_filename,test_size=20)
        # accuracy += train_246_fcn.main(brain_map=brain_map,data_type=data_type,pre_dir=target_sMRI_dir,num_batches=1024*10+1,test_size=20)
        # 输出测试准确率
        # output = 'iter {}:{}'.format(step, one_accuracy)
        # print(output)
        # f.write(output + '\n')
        output = 'iter {}:\n model:{}\n SVM_model:{}\nSVM_ensemble_modle:{}\nensemble_model:{}'.format(step,BGRU_ACC,BGRU_SVM_ACC,ensemble_SVM_ACC,ensemble_ACC)
        print(output)
        f.write(output+'\n')
    f.close()

def xtest(_):
    train_list = list(range(10))
    test_list = list(range(20))
    for remove in test_list:
        print(remove)
        if remove in train_list:
            test_list = np.delete(test_list, remove)
        print(test_list)
if __name__ == '__main__':
    tf.app.run(train)
