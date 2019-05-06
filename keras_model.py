import keras
import tensorflow as tf
import evaluation
from fmri_data import fMRI_data
from voxnet import VoxNet
from config import cfg
import numpy as np
import toolbox
import os
from keras.layers import *
import nibabel as nib
from sklearn import svm
import shutil
def FCNs(input_shape=[80,50],nb_class=2):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(128,8,padding='same',input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv1D(256,5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(128, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(nb_class))
    model.add(keras.layers.Activation('softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model

def CNN_3D(input_shape=[32,32,32,1],nb_class = 2):
    model = keras.Sequential()
    model.add(keras.layers.Conv3D(8,3,padding='same',input_shape=input_shape,kernel_regularizer=keras.regularizers.l2(0.01),kernel_initializer=keras.initializers.glorot_uniform(seed=None)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv3D(8,2,strides=[2,2,2],padding='valid',kernel_regularizer=keras.regularizers.l2(0.01),kernel_initializer=keras.initializers.glorot_uniform(seed=None)))

    model.add(keras.layers.Conv3D(8, 3, padding='same', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv3D(8, 2, strides=[2, 2, 2], padding='valid',kernel_regularizer=keras.regularizers.l2(0.01),kernel_initializer=keras.initializers.glorot_uniform(seed=None)))

    model.add(keras.layers.Conv3D(8, 3, padding='same', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv3D(8, 2, strides=[2, 2, 2], padding='valid',kernel_regularizer=keras.regularizers.l2(0.01),kernel_initializer=keras.initializers.glorot_uniform(seed=None)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(300,kernel_regularizer=keras.regularizers.l2(0.01),kernel_initializer=keras.initializers.glorot_uniform(seed=None)))
    model.add(Dropout(0.2))
    model.add(keras.layers.Dense(100,kernel_regularizer=keras.regularizers.l2(0.01),kernel_initializer=keras.initializers.glorot_uniform(seed=None),name='CNN_fc2'))
    model.add(Dropout(0.2))
    model.add(keras.layers.Dense(nb_class,kernel_regularizer=keras.regularizers.l2(0.01),kernel_initializer=keras.initializers.glorot_uniform(seed=None)))
    model.add(keras.layers.Activation('softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

# CNN_3D()
def GRUs(input_shape=[80,100],nb_class = 2):

    # # keras.layers.GRU(300,go_backwards=True,return_sequences=True)
    # model_left = keras.Sequential()
    # model_left.add(keras.layers.GRU(units=300,return_sequences=True,input_shape=[5,300]))
    # model_left.add(keras.layers.GRU(units=200,return_sequences=True))
    # model_left.add(keras.layers.Dropout(0.2))
    # model_left.add(keras.layers.GRU(units=50,return_sequences=True))
    # model_right = keras.Sequential()
    # model_right.add(keras.layers.GRU(units=300, return_sequences=True, go_backwards=True,input_shape=[5, 300]))
    # model_right.add(keras.layers.GRU(units=200, return_sequences=True))
    # model_right.add(keras.layers.Dropout(0.2))
    # model_right.add(keras.layers.GRU(units=50,return_sequences=False))
    #
    # model = keras.Sequential()
    # model.add()
    # model.add(keras.layers.Dense(50))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.Dense(2))
    # model.add(keras.layers.Activation('softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='adam',
    #               metrics=['accuracy'])
    a = Input(shape=input_shape,name='BGRU_input')
    b = Masking(mask_value=-1)(a)
    left = GRU(units=80,return_sequences=True,kernel_initializer=keras.initializers.glorot_uniform(seed=None))(b)
    # left = Dropout(0.2,name='BGRU_Dropout_1')(left)
    left = GRU(units=50,return_sequences=True,kernel_initializer=keras.initializers.glorot_uniform(seed=None))(left)
    left = Dropout(0.7,name='BGRU_Dropout_2')(left)
    left = GRU(units=25,return_sequences=False,kernel_initializer=keras.initializers.glorot_uniform(seed=None))(left)
    right = GRU(units=80, return_sequences=True,go_backwards=True,kernel_initializer=keras.initializers.glorot_uniform(seed=None),kernel_regularizer=keras.regularizers.l2(0.01))(b)
    # right = Dropout(0.2,name='BGRU_Dropout_3')(right)
    right = GRU(units=50, return_sequences=True,kernel_initializer=keras.initializers.glorot_uniform(seed=None))(right)
    right = Dropout(0.7,name='BGRU_Dropout_4')(right)
    right = GRU(units=25, return_sequences=False,kernel_initializer=keras.initializers.glorot_uniform(seed=None))(right)
    fc = concatenate([left,right],axis=-1,name='fc_1')
    fc = Activation('relu')(fc)
    # fc = Dense(50)(fc)
    # output = Dropout(0.2)(fc)
    output = Dense(nb_class,kernel_initializer=keras.initializers.glorot_uniform(seed=None),name='fc_2',kernel_regularizer=keras.regularizers.l2(0.01))(fc)
    output = Activation('softmax')(output)
    model = keras.Model(inputs=a,outputs=output)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model
GRUs()
def train_fcn(cross_epoch = 0,data_index=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data',
         num_batches = 256*5,voxnet_point=None,test_size = 6,brain_map=[217]):
    tf.reset_default_graph()
    #####超参##########
    time_dim = 20
    batch_size = 8
    #####################
    dataset = fMRI_data(data_type, data_index=data_index, varbass=False, dir=pre_dir)
    fcn = FCNs()
    input_shape = [None,32,32,32,1]
    voxnet = VoxNet(input_shape=input_shape,voxnet_type='cut')
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        voxnet.npz_saver.restore(session,voxnet_point)
        fcn.fit_generator(dataset.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size = batch_size,_batch_mode='oversampling',_mode='train'),steps_per_epoch=8,epochs=num_batches,
                          validation_data=dataset.test.random_sampling.get_time_batch(session,voxnet,cut_shape,time_dim=time_dim,batch_size=1,_batch_mode='random',_mode='test'),validation_steps=test_size,
                          callbacks=[toolbox.EarlyStoppingByACC('loss',0.3,patience=10)])
        #预测
        data_iter = iter(dataset.test.random_sampling.get_time_batch(session, voxnet, cut_shape, time_dim=time_dim, batch_size=1))

            # for i in range(num_batches / 16):
            #     fcn.fit_generator(dataset.train.oversampling.get_time_batch(session, voxnet, cut_shape, time_dim=time_dim,
            #                                                                 batch_size=batch_size), steps_per_epoch=8,
            #                       epochs=num_batches)
        test_evaluation = evaluation.evaluation()
        for i in range(test_size):
            vosx,onehot = data_iter.__next__()
            prediction = fcn.predict_on_batch(vosx)
            test_evaluation += evaluation.evaluation(y_true=np.argmax(onehot,axis=1),y_predict=np.argmax(prediction,axis=1))
            print(test_evaluation)
        filepath = os.path.join(cfg.fcn_checkpoint_dir,'train_'+str(cross_epoch)+'.h5')
        fcn.save(filepath)
    return test_evaluation

def train_GRUs(cross_epoch = 0,data_index=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data',
         num_batches = 256*5,voxnet_point=None,test_size = 6,brain_map=[217],f_handle=None):
    #清楚keras后台数据
    keras.backend.clear_session()
    # #####超参##########
    time_dim = 80
    batch_size = 50
    g1 = tf.Graph()
    g2 = tf.Graph()
    # keras.optimizers.Adam()
    # #####################
    # #加载模型#
    xyz = 32
    # input_shape = [xyz, xyz, xyz, 1]
    # inputs = keras.layers.Input(input_shape)

    _3D_CNN = keras.models.load_model(voxnet_point)
    # _3D_CNN.predict(np.zeros([1,32,32,32,1]),1)
    # for i in _3D_CNN.layers:
    #     print(i.name)
    layer_name = 'CNN_fc2'
    feature_generator = keras.Model(inputs=_3D_CNN.layers[0].input, outputs=_3D_CNN.get_layer(layer_name).output)
    # sample = np.zeros([1, xyz, xyz, xyz, 1])
    # y = feature_generator.predict_on_batch(sample)
    # print(y)



    dataset = fMRI_data(data_type, data_index=data_index, varbass=False, dir=pre_dir,model=feature_generator,cut_shape=cut_shape)
    grus = GRUs(input_shape=[time_dim, 100], nb_class=2)
    # feature_generator = keras.Sequential()
    # feature_generator.get_layer()
    # voxnet = VoxNet(input_shape=input_shape, voxnet_type='cut')
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     voxnet.npz_saver.restore(session, voxnet_point)
    logs_path = os.path.join(cfg.fcn_checkpoint_dir, 'train_' + str(cross_epoch))
    if os.path.isdir(logs_path):
        shutil.rmtree(logs_path)
    grus.fit_generator(dataset.get_time_batch(cut_shape= cut_shape, time_dim=time_dim, batch_size=batch_size,_batch_mode='oversampling', _mode='train'), steps_per_epoch=8,
                          epochs=num_batches,
                          validation_data=dataset.get_time_batch(cut_shape= cut_shape, time_dim=time_dim, batch_size=1,_batch_mode='random',_mode='test',flag=1),
                          validation_steps=test_size,
                          callbacks=[toolbox.EarlyStoppingByACC('acc', 0.85, patience=10),keras.callbacks.TensorBoard(log_dir=logs_path)])
    # 预测
    #构造循环迭代器，顺序不打乱的
    train_iter = iter(dataset.get_time_batch(cut_shape=cut_shape, time_dim=time_dim, batch_size=1,_batch_mode='random',_mode='train',flag=1))
    test_iter = iter(dataset.get_time_batch(cut_shape=cut_shape, time_dim=time_dim, batch_size=1,_batch_mode='random',_mode='test',flag=1))

        # for i in range(num_batches / 16):
        #     fcn.fit_generator(dataset.train.oversampling.get_time_batch(session, voxnet, cut_shape, time_dim=time_dim,
        #                                                                 batch_size=batch_size), steps_per_epoch=8,
        #                       epochs=num_batches)
    test_evaluation = evaluation.evaluation()
    print('test evaluation:')
    for i in range(test_size):
        vosx, onehot = test_iter.__next__()
        prediction = grus.predict_on_batch(vosx)
        test_evaluation += evaluation.evaluation(y_true=np.argmax(onehot, axis=1),
                                                 y_predict=np.argmax(prediction, axis=1))
        print(test_evaluation)



    train_len = 0
    for i in data_type:
        train_len += len(data_index[i]['train'])

    train_evaluation = evaluation.evaluation()
    for i in range(test_size):
        vosx, onehot = train_iter.__next__()
        prediction = grus.predict_on_batch(vosx)
        train_evaluation += evaluation.evaluation(y_true=np.argmax(onehot, axis=1),
                                                 y_predict=np.argmax(prediction, axis=1))
        print(test_evaluation)

    # 利用svm作为最后的分类器
    print('svm evaluation:')
    # 构造数据集
    #BGRU特征提取模型
    BGRU_feature = keras.Model(inputs=grus.get_layer('BGRU_input').input,outputs=grus.get_layer('fc_1').output)
    train_data = -1
    train_label = -1
    for i in range(train_len):
        vosx,onehot = train_iter.__next__()
        feature = BGRU_feature.predict_on_batch(vosx)
        y_true = np.argmax(onehot,axis=1)
        if isinstance(train_data,int):
            train_data = feature
            train_label = y_true
        else:
            train_data = np.vstack((train_data,feature))
            train_label = np.hstack((train_label,y_true))
    clf = svm.SVC(C=1.0,kernel='rbf',gamma='auto')
    clf.fit(train_data,train_label)
    predict = clf.predict(train_data)
    svm_train = evaluation.evaluation(y_true=train_label,y_predict=predict)
    print('svm_train:')
    print(svm_train)

    test_data = -1
    test_label = -1
    for i in range(test_size):
        vosx,onehot = test_iter.__next__()
        feature = BGRU_feature.predict_on_batch(vosx)
        y_true = np.argmax(onehot,axis=1)
        if isinstance(test_data,int):
            test_data = feature
            test_label = y_true
        else:
            test_data = np.vstack((test_data,feature))
            test_label = np.hstack((test_label,y_true))
    predict = clf.predict(test_data)
    svm_test = evaluation.evaluation(y_true=test_label, y_predict=predict)
    print('svm_test:')
    print(svm_test)

    if f_handle:
        f_handle.write('svm_train:\n')
        f_handle.write(str(svm_train)+'\n')
        f_handle.write('svm_test:\n')
        f_handle.write(str(svm_test)+'\n')
        f_handle.write('train_evaluation\n')
        f_handle.write(str(train_evaluation) + '\n')
        f_handle.write('test_evaluation\n')
        f_handle.write(str(test_evaluation)+'\n')

    if not os.path.exists(cfg.fcn_checkpoint_dir):
        os.makedirs(cfg.fcn_checkpoint_dir)
        filepath = os.path.join(cfg.fcn_checkpoint_dir, 'train_' + str(cross_epoch)+'.h5')
        grus.save(filepath=filepath)

    return test_evaluation,svm_test
    # return evaluation.evaluation()


def train_3DCNN(cross_epoch = 0,data_index=None,brain_map=None,cut_shape=None,data_type=['MCIc','MCInc'],pre_dir='/home/anzeng/rhb/fmri_data', num_batches = 512*5+1,test_size=6):
    keras.backend.clear_session()
    batch_size = 50
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
    # xyz = 32
    logs_path = os.path.join(cfg.voxnet_checkpoint_dir,'train_'+str(cross_epoch))
    if os.path.isdir(logs_path):
        shutil.rmtree(logs_path)
    os.makedirs(logs_path)
    model = CNN_3D()
    dataset = fMRI_data(data_type, data_index=data_index,dir=pre_dir, batch_mode='random', varbass=cfg.varbass)
    model.fit_generator(dataset.get_smri_batch(cut_shape,batch_size,_batch_mode='oversampling',_mode='train'), steps_per_epoch=8,
                       epochs=num_batches,
                       validation_data=dataset.get_smri_batch(cut_shape,batch_size=20,_batch_mode='random',_mode='test'),
                       validation_steps=test_size,
                       callbacks=[toolbox.EarlyStoppingByACC('acc', 0.85, patience=10),keras.callbacks.TensorBoard(log_dir=logs_path)])

    if not os.path.exists(cfg.voxnet_checkpoint_dir):
        os.makedirs(cfg.voxnet_checkpoint_dir)
    filepath = os.path.join(cfg.voxnet_checkpoint_dir, 'train_' + str(cross_epoch) + '.h5')
    model.save(filepath=filepath)
    # del model
    # model = keras.models.load_model(filepath)
    return filepath
# _3DCNN()
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras import backend as K
# from keras.utils.vis_utils import plot_model
# batch_size = 128
# nb_classes = 10
# nb_epoch = 1
#
# # 输入数据的维度
# img_rows, img_cols = 28, 28
# # 使用的卷积滤波器的数量
# nb_filters = 32
# # 用于 max pooling 的池化面积
# pool_size = (2, 2)
# # 卷积核的尺寸
# kernel_size = (3, 3)
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# #同样要reshape,只是现在图片是三维矩阵
# X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# input_shape = (img_rows, img_cols, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
# model = Sequential()
#
# model.add(Convolution2D(nb_filters, kernel_size,strides=(1, 1),
#                         padding='valid',
#                         input_shape=input_shape))#用作第一层时，需要输入input_shape参数
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, kernel_size))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.25))#Dense()的前面要减少连接点，防止过拟合，故通常要Dropout层或池化层
# model.add(Flatten())#Dense()层的输入通常是2D张量，故应使用Flatten层或全局平均池化
# model.add(Dense(128))
# model.add(Activation('relu'))#Dense( )层的后面通常要加非线性化函数
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))#分类
# model.summary()
# plot_model(model, to_file='model-cnn.png')
# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',metrics=['acc'])
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
# model.fit(X_train, Y_train, validation_split=0.33, nb_epoch=1, batch_size=128)
# # print(model.history)
# # model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
