from fmri_data import fMRI_data
from voxnet import VoxNet
import tensorflow as tf
from evaluation import evaluation
import numpy as np
import keras
from sklearn import svm
import os
from config import cfg
#标签集成
def ensemble_label(label,label_len):
    predict_label = 0
    predict_cnt = 0
    label = label.astype(np.int32)
    cnt = np.bincount(label)
    # print(cnt)
    for i in range(label_len):
        if i < len(cnt) and cnt[i] > predict_cnt:
            predict_cnt = cnt[i]
            predict_label = i
    return predict_label

def get_label(model,img,label,cut_shape,true_shape):
    # p = dict()
    # p['predict'] = voxnet[-1]
    # p['predict'] = tf.argmax(p['predict'], axis=1)
    xyz = 32
    time_dim = img.shape[0]
    # 只取奇数张smri
    if time_dim % 2 == 0:
        time_dim -= 1
    new_feature = img[0:time_dim, cut_shape[0]:cut_shape[1] + 1,
                  cut_shape[2]:cut_shape[3] + 1,
                  cut_shape[4]:cut_shape[5] + 1]
    # 调整形状
    new_shape = [new_feature.shape[0], true_shape[0], true_shape[1], true_shape[2], 1]
    new_feature = np.reshape(new_feature, new_shape)
    # 计算图像的坐标开始位置
    start_x = (xyz - true_shape[0]) // 2
    start_y = (xyz - true_shape[1]) // 2
    start_z = (xyz - true_shape[2]) // 2
    voxs = np.zeros([new_feature.shape[0], xyz, xyz, xyz, 1], np.float32)
    voxs[0:new_feature.shape[0], start_x:start_x + true_shape[0],
    start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = new_feature
    all_predict =  model.predict_on_batch(voxs)
    # all_predict = np.argmax(all_predict,axis=1)
    # 调整形状,形状为[80,mri维度，1]
    # new_shape = [min(time_dim, img.shape[0]), shape[1], shape[2], shape[3], 1]
    # new_feature = np.reshape(new_feature, new_shape)
    # all_predict = sess.run(p['predict'],
    #                        feed_dict={voxnet[0]: voxs, voxnet.keep_prob: 1.0, voxnet.training: False})
    label = np.ones(time_dim) * label
    return all_predict,label

def emsemble(cross_epoch = 0,data_index=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data',
         num_batches = 256*5,voxnet_point=None,test_size = 6,brain_map=[217],f_handle = None):
    # tf.reset_default_graph()
    keras.backend.clear_session()
    dataset = fMRI_data(data_type, data_index=data_index, varbass=False, dir=pre_dir)
    # xyz = 32
    # input_shape = [None, xyz, xyz, xyz, 1]
    # voxnet = VoxNet(input_shape=input_shape, voxnet_type='cut')

    true_shape = []
    for x in range(0, len(cut_shape), 2):
        true_shape.append(cut_shape[x + 1] - cut_shape[x] + 1)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     voxnet.npz_saver.restore(sess,voxnet_point)
    #加载模型
    model = keras.models.load_model(voxnet_point)
    print('train_acc')
    train_fmri_evaluation = evaluation()
    train_smri_evaluation = evaluation()
    train_iter = iter(dataset.get_fmri('train')).__next__
    for i in range(100):
        img,label,_ = train_iter()
        predict,y_true = get_label(model,img,label,cut_shape,true_shape)
        predict = np.argmax(predict,axis=1)
        train_smri_evaluation += evaluation(y_predict=predict,y_true=y_true)
        if i %10 == 0 and i > 0:
            print(train_smri_evaluation)
        y_predict = ensemble_label(predict,2)
        train_fmri_evaluation += evaluation(y_predict = [y_predict],y_true=[label])
    print(train_fmri_evaluation)
    print('test_acc')
    test_fmri_evaluation = evaluation()
    test_smri_evaluation = evaluation()
    test_iter = iter(dataset.get_fmri('test')).__next__
    for i in range(test_size):
        img, label,filename = test_iter()
        predict, y_true = get_label(model, img, label, cut_shape, true_shape)
        predict = np.argmax(predict,axis=1)
        test_smri_evaluation_one = evaluation(y_predict=predict, y_true=y_true)
        test_smri_evaluation += test_smri_evaluation_one
        print(test_smri_evaluation_one)
        print(test_smri_evaluation)
        y_predict = ensemble_label(predict,2)
        test_fmri_evaluation += evaluation(y_predict=[y_predict], y_true=[label])
        print(y_predict,label,test_fmri_evaluation)
        # if y_predict != label:
        #     print(filename)
        #     f_handle.write(filename+'\n')
    if f_handle:
        f_handle.write('ensemble train:\n')
        f_handle.write(str(train_fmri_evaluation) + '\n')
        f_handle.write('ensemble test:\n')
        f_handle.write(str(test_fmri_evaluation) + '\n')
    return test_fmri_evaluation

def svm_emsemble(cross_epoch = 0,data_index=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data',
         num_batches = 256*5,voxnet_point=None,test_size = 6,brain_map=[217],f_handle = None):
    keras.backend.clear_session()
    dataset = fMRI_data(data_type, data_index=data_index, varbass=False, dir=pre_dir)

    true_shape = []
    for x in range(0, len(cut_shape), 2):
        true_shape.append(cut_shape[x + 1] - cut_shape[x] + 1)
    # 加载模型
    model = keras.models.load_model(voxnet_point)
    layer_name = 'CNN_fc2'
    model = keras.Model(inputs=model.layers[0].input, outputs=model.get_layer(layer_name).output)
    #加载模型
    train_len = 0
    for i in data_type:
        train_len += len(data_index[i]['train'])

    # print('train_acc')
    # train_fmri_evaluation = evaluation()
    # train_smri_evaluation = evaluation()
    train_iter = iter(dataset.get_fmri('train')).__next__
    train_one_len = [0]
    #训练数据
    train_data = -1
    train_label = -1
    for i in range(train_len):
        img, label, _ = train_iter()
        predict, y_true = get_label(model, img, label, cut_shape, true_shape)
        if isinstance(train_data,int):
            train_data = predict
            train_label = y_true
        else:
            train_data = np.vstack((train_data,predict))
            train_label = np.hstack((train_label,y_true))
        train_one_len.append(train_one_len[-1]+predict.shape[0])
        # train_smri_evaluation += evaluation(y_predict=predict, y_true=y_true)
        # y_predict = ensemble_label(predict, 2)
        # train_fmri_evaluation += evaluation(y_predict=[y_predict], y_true=[label])
    # print(train_fmri_evaluation)
    clf = svm.SVC(C=1.0,kernel='rbf',gamma='auto')
    clf.fit(train_data,train_label)
    train_evaluation = evaluation()
    for i in range(1,len(train_one_len),1):
        predict = clf.predict(train_data[train_one_len[i-1]:train_one_len[i]])
        # print(predict)
        y_predict = ensemble_label(predict, 2)
        y_true = train_label[train_one_len[i-1]]
        train_evaluation += evaluation(y_predict=[y_predict], y_true=[y_true])
    # predict = clf.predict(train_data)
    # train_evaluation = evaluation(y_true = train_label,y_predict = predict)
    print('svm ensemble train:')
    print(train_evaluation)

    test_evaluation = evaluation()
    # test_smri_evaluation = evaluation()
    print('svm ensemble test')
    test_iter = iter(dataset.get_fmri('test')).__next__
    for i in range(test_size):
        img, label, filename = test_iter()
        predict, y_true = get_label(model, img, label, cut_shape, true_shape)
        # test_smri_evaluation_one = evaluation(y_predict=predict, y_true=y_true)
        # test_smri_evaluation += test_smri_evaluation_one
        # print(test_smri_evaluation_one)
        # print(test_smri_evaluation)
        predict = clf.predict(predict)
        y_predict = ensemble_label(predict, 2)
        test_evaluation += evaluation(y_predict=[y_predict], y_true=[label])
        print(y_predict, label, test_evaluation)
        # if y_predict != label:
        #     print(filename)
        #     f_handle.write(filename+'\n')
    if f_handle:
        f_handle.write('svm ensemble train:\n')
        f_handle.write(str(train_evaluation)+'\n')
        f_handle.write('svm ensemble test:\n')
        f_handle.write(str(test_evaluation)+'\n')
    return test_evaluation

