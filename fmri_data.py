import nibabel as nib
import os
import random
import numpy as np
import tensorflow as tf
import time

from config import cfg
from voxnet import VoxNet

class fMRI_data(object):

    #不同类别赋予不同的权值
    def __init__(self, data_type=['AD','NC'],data_index=None,batch_size=None,batch_mode = 'oversampling',varbass = False,dir="/home/anzeng/rhb/fmri_data",model = None,cut_shape = None):
        #data_index={'data_type':{'train':[],'test':[]}}
        class MRI(object):
            def __init__(self, fi, label, category):
                #fi:路径名,label:标签,category:类别
                self._fi = fi
                self._label = label
                self._category = category

            @property
            def mri(self):
                # x = self._zf.read(self._fi)
                # print(fi)
                return np.load(self._fi)

            @property
            def label(self):
                return self._label

            @property
            def category(self):
                return self._category


            @property
            def filename(self):
                return self._fi.filename.split('/')[-1]

            def save(self, f=None):
                self.filename if f is None else f
                np.save(f, self.mri)

        ############初始化数据集信息######################
        self._batch_size = batch_size
        self._varbass = varbass #是否输出调试信息
        self._mode = 'train'
        self._batch_mode = batch_mode #采样模式 oversampling:过取样,random:随机取样
        self._data_type=data_type
        self._dir = dir  #地址索引
        self._iters = {'oversampling':{'train':{},'test':{}},'random':{}}
        self._data = {'train': [], 'test': []}
        self._data_len = {'train':{},'test':{}}
        self._model = model

        ###################################################

        ########迭代器############
        def get_random_iter(mode,_batch_mode,data_type = None,):
            # print(mode, data_type)
            while 1:
                if _batch_mode == 'oversampling':
                    seq = self._data_len[mode][data_type]
                if _batch_mode == 'random':
                    seq = [0,len(self._data[mode])]
                order = np.arange(seq[0],seq[1],1)
                # print('order:',order)
                np.random.shuffle(order)
                for i in order:
                    yield i
        print('Setting up ' +str(self._data_type)+'database... ')

        #加载数据，模型:标签，文件路径
        true_shape = []  # 实际大小
        # flag = 0
        if self._model:
            for x in range(0, len(cut_shape), 2):
                true_shape.append(cut_shape[x + 1] - cut_shape[x] + 1)
        p=dict()
        p['train_start'] = 0
        p['test_start'] = 0
        if data_index==None:
            for classification in self._data_type:
                now_dir = os.path.join(self._dir,classification)
                for mode in ['train','test']:
                    now_dir_x = os.path.join(now_dir,mode)
                    #数据集类别的区间
                    variables_name = mode + '_start'
                    self._data_len[mode][classification]=[p[variables_name],p[variables_name]+len(os.listdir(now_dir_x))]
                    p[variables_name] += len(os.listdir(now_dir_x))
                    for i in os.listdir(now_dir_x):
                        category = classification
                        train_or_test=mode
                        filename = os.path.join(now_dir_x,i)
                        self._data[train_or_test].append((category,filename))
        else:
            for classification in self._data_type:
                now_dir = os.path.join(self._dir,classification)
                for mode in ['train','test']:
                    lists = os.listdir(now_dir)
                    #数据集类别的区间
                    variables_name = mode + '_start'
                    self._data_len[mode][classification]=[p[variables_name],p[variables_name]+len(data_index[classification][mode])]
                    p[variables_name] += len(data_index[classification][mode])
                    for i in data_index[classification][mode]:
                        category = classification
                        train_or_test=mode
                        filename = os.path.join(now_dir,lists[i])
                        if self._model:
                            xyz = 32
                            img = np.load(filename)
                            new_feature = img[0:img.shape[0], cut_shape[0]:cut_shape[1] + 1,
                                          cut_shape[2]:cut_shape[3] + 1,
                                          cut_shape[4]:cut_shape[5] + 1]
                            # 调整形状
                            new_shape = [new_feature.shape[0], true_shape[0], true_shape[1], true_shape[2], 1]
                            new_feature = np.reshape(new_feature, new_shape)
                            start_x = (xyz - true_shape[0]) // 2
                            start_y = (xyz - true_shape[1]) // 2
                            start_z = (xyz - true_shape[2]) // 2
                            voxs = np.zeros([new_feature.shape[0], xyz, xyz, xyz, 1], np.float32)
                            voxs[0:new_feature.shape[0], start_x:start_x + true_shape[0],
                            start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = new_feature
                            # voxs = np.reshape(img,[-1,61,73,61,1])
                            time_feature = model.predict_on_batch(voxs)
                            self._data[train_or_test].append((category,time_feature))
                        else:
                            self._data[train_or_test].append((category,filename))
        # print(self._data_len)
        #标签制作
        categories = sorted(list(set(c for c, i in self._data['test'])))
        categories = dict(zip(categories, range(len(categories))))
        print(categories)
        for k in self._data:
            self._data[k] = [MRI(i, categories[c], c) for c, i in self._data[k]]
            # self._batch_mode='random'
            self._iters['random'][k] = iter(get_random_iter(k,'random'))
            #数据种类
            for x in self._data_type:
                # self._batch_mode='oversampling'
                self._iters['oversampling'][k][x] = iter(get_random_iter(k,'oversampling',x))
        self.categories = categories.keys()
        print(str(self._data_type) + 'database setup complete!')


    @property
    def num_categories(self):
        return len(self.categories)

    @property
    def train(self):
        self._mode = 'train'
        return self

    @property
    def test(self):
        self._mode = 'test'
        return self

    @property
    def oversampling(self):
        self._batch_mode = 'oversampling'
        return self

    @property
    def random_sampling(self):
        self._batch_mode='random'
        return self
    @property
    def data(self):
        return self._data[self._mode]

    def __len__(self):
        return len(self._data[self._mode])

    def get_fmri_brain(self,cut_shape,batch_size,time_dim):
        xyz = 32 #x,y,z大小
        shape=[batch_size*time_dim,xyz,xyz,xyz,1] #输入的形状大小
        true_shape = [] #实际大小
        for x in range(0,len(cut_shape),2):
            true_shape.append(cut_shape[x + 1] - cut_shape[x] + 1)
        if self._batch_mode == 'random':
            shape[0] = shape[0] * 4
        voxs = np.zeros(shape,dtype=np.float32)
        one_hots = np.zeros([batch_size,self.num_categories],dtype=np.float32)
        data = self._data[self._mode]
        total = 0
        #获取两个不同的样本
        if self._batch_mode == 'oversampling':
        #当取样模式为过取样时,batch_size表示每一个样本取样的个数
            for i in self._data_type:
                for j in range(batch_size//2):
                    index = self._iters[self._batch_mode][self._mode][i].__next__()
                    # 加载图片
                    img = nib.load(data[index]._fi)
                    img = img.get_fdata()
                    img = np.transpose(img, [3, 0, 1, 2])
                    if self._varbass:
                        print(img.shape)
                    new_feature = img[0:min(time_dim,img.shape[0]), cut_shape[0]:cut_shape[1] + 1, cut_shape[2]:cut_shape[3] + 1,
                                  cut_shape[4]:cut_shape[5] + 1]
                    #数据归一化
                    min_vox = np.min(new_feature)
                    max_vox = np.max(new_feature)
                    new_feature = (new_feature - min_vox)/(max_vox - min_vox)
                    # 调整形状,形状为[80,mri维度，1]
                    new_shape = [min(time_dim,img.shape[0]), true_shape[0], true_shape[1], true_shape[2], 1]
                    new_feature = np.reshape(new_feature, new_shape)
                    #数据增强
                    axis = np.random.choice([0,1,2,3],1)
                    if axis[0] > 0:
                        new_feature = np.flip(new_feature,axis=axis[0])
                    start_t = (time_dim - new_shape[0])//2
                    start_x = (xyz - true_shape[0])//2
                    start_y = (xyz - true_shape[1])//2
                    start_z = (xyz - true_shape[2])//2
                    voxs[total+start_t:total + start_t + new_shape[0],start_x:start_x+true_shape[0],start_y:start_y+true_shape[1],start_z:start_z+true_shape[2],0:1] = new_feature
                    one_hots[int(total/time_dim), data[index].label] = 1
                    total = total + time_dim
                    if self._varbass:
                        print(total)
        if self._batch_mode == 'random':
            for bs in range(batch_size):
                index = self._iters[self._batch_mode][self._mode].__next__()
                # 加载图片
                img = nib.load(data[index]._fi)
                img = img.get_fdata()
                img = np.transpose(img, [3, 0, 1, 2])
                if self._varbass:
                    print(img.shape)
                new_feature = img[0:min(time_dim, img.shape[0]), cut_shape[0]:cut_shape[1] + 1,
                              cut_shape[2]:cut_shape[3] + 1,
                              cut_shape[4]:cut_shape[5] + 1]
                # 调整形状
                new_shape = [min(time_dim,img.shape[0]), true_shape[0], true_shape[1], true_shape[2], 1]
                new_feature = np.reshape(new_feature, new_shape)
                # 数据归一化
                min_vox = np.min(new_feature)
                max_vox = np.max(new_feature)
                new_feature = (new_feature - min_vox) / (max_vox - min_vox)
                start_t = (time_dim - new_shape[0]) // 2
                start_x = (xyz - true_shape[0]) // 2
                start_y = (xyz - true_shape[1]) // 2
                start_z = (xyz - true_shape[2]) // 2
                voxs[total + start_t:total +start_t+ new_shape[0], start_x:start_x + true_shape[0],
                start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = new_feature
                total = total + time_dim
                for replace in range(3):
                    flip_feature = np.flip(new_feature,axis=replace+1)
                    voxs[total + start_t:total + start_t + new_shape[0], start_x:start_x + true_shape[0],start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = flip_feature
                    total = total + time_dim
                one_hots[bs, data[index].label] = 1
                if self._varbass:
                    print(total)
        # print(one_hots)
        return voxs[0:total],one_hots

    def get_smri_batch(self,cut_shape,batch_size,_batch_mode=None,_mode = None):
        xyz = 32  # x,y,z大小
        # shape = [batch_size,61,73,61,1]
        shape = [batch_size,xyz, xyz, xyz, 1]  # 输入的形状大小
        true_shape = []  # 实际大小
        # flag = 0
        for x in range(0, len(cut_shape), 2):
            true_shape.append(cut_shape[x + 1] - cut_shape[x] + 1)
        if _batch_mode == None:
            _batch_mode = self._batch_mode
        if _mode == None:
            _mode = self._mode
        if _mode == 'test':
            _iter = iter(self.data_iter('test')).__next__
        # if self._batch_mode == 'random':
        #     shape[0] = shape[0] * 4
        while 1:
            voxs = np.zeros(shape, dtype=np.float32)
            one_hots = np.zeros([batch_size, self.num_categories], dtype=np.float32)
            data = self._data[_mode]
            total = 0
            self._batch_mode = _batch_mode
            self._mode = _mode
            # 获取两个不同的样本
            if _batch_mode == 'oversampling':
                # 当取样模式为过取样时,batch_size表示每一个样本取样的个数
                for i in self._data_type:
                    for j in range(batch_size // 2):
                        index = self._iters[_batch_mode][_mode][i].__next__()
                        # 加载图片
                        img = np.load(data[index]._fi)
                        # if flag == 0:
                        #     flag = 1
                        #     print(np.max(img) - np.min(img))
                        # img = img.get_fdata()
                        # img = np.transpose(img, [3, 0, 1, 2])
                        if self._varbass:
                            print(img.shape)
                        new_feature = img[cut_shape[0]:cut_shape[1] + 1,
                                      cut_shape[2]:cut_shape[3] + 1,
                                      cut_shape[4]:cut_shape[5] + 1]
                        new_shape = [true_shape[0], true_shape[1], true_shape[2], 1]

                        # new_feature = img
                        # new_shape = [61, 73, 61, 1]
                        new_feature = np.reshape(new_feature, new_shape)
                        # # 数据增强
                        # axis = np.random.choice([0, 1, 2, 3], 1)
                        # if axis[0] > 0:
                        #     new_feature = np.flip(new_feature, axis=axis[0])

                        start_x = (xyz - true_shape[0]) // 2
                        start_y = (xyz - true_shape[1]) // 2
                        start_z = (xyz - true_shape[2]) // 2
                        voxs[total][start_x:start_x + true_shape[0],
                        start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = new_feature
                        # voxs[total] = new_feature
                        one_hots[total][data[index].label] = 1
                        total += 1
                        if self._varbass:
                            print(total)
            if _batch_mode == 'random':
                for bs in range(batch_size):
                    index = self._iters[_batch_mode][_mode].__next__()
                    # 加载图片
                    img = np.load(data[index]._fi)
                    # img = img.get_fdata()
                    # img = np.transpose(img, [3, 0, 1, 2])
                    if self._varbass:
                        print(img.shape)
                    new_feature = img[cut_shape[0]:cut_shape[1] + 1,
                                  cut_shape[2]:cut_shape[3] + 1,
                                  cut_shape[4]:cut_shape[5] + 1]
                    # # 调整形状
                    new_shape = [true_shape[0], true_shape[1], true_shape[2], 1]
                    new_feature = np.reshape(new_feature, new_shape)
                    start_x = (xyz - true_shape[0]) // 2
                    start_y = (xyz - true_shape[1]) // 2
                    start_z = (xyz - true_shape[2]) // 2
                    voxs[total][start_x:start_x + true_shape[0],
                    start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = new_feature

                    # new_feature = img
                    # new_shape = [61, 73, 61, 1]
                    # new_feature = np.reshape(new_feature, new_shape)
                    # voxs[total] = new_feature
                    total = total + 1
                    one_hots[bs, data[index].label] = 1
                    if self._varbass:
                        print(total)
            # print(one_hots)
            yield voxs, np.reshape(voxs,[-1,xyz**3])


    def get_batch(self, batch_size=None):
        rn = random.randint
        bs = batch_size if batch_size is not None else self._batch_size
        bs = bs if bs is not None else 16
        voxs = np.zeros([bs, 61, 73, 61, 1], dtype=np.float32)
        one_hots = np.zeros([bs, self.num_categories], dtype=np.float32)
        data = self._data[self._mode]
        next_int = self._iters[self._mode].__next__
        P,N = 0,0
        sum = 0
        #令两个类别的数据数量相同
        if self._mode == 'train' and self._batch_mode=='oversampling':
            while sum < bs:
                index = next_int()
                if data[index].label == 0 and P < batch_size//2:
                    v = data[index]
                    d = v.mri.reshape([61, 73, 61, 1])
                    for axis in 0, 1, 2:
                        if rn(0, 1):
                            d = np.flip(d, axis)
                    voxs[sum] = d
                    one_hots[sum][v.label] = 1
                    P += 1
                    sum += 1
                if data[index].label == 1 and N < batch_size//2:
                    v = data[index]
                    d = v.mri.reshape([61, 73, 61, 1])
                    for axis in 0, 1, 2:
                        if rn(0, 1):
                            d = np.flip(d, axis)
                    voxs[sum] = d
                    one_hots[sum][v.label] = 1
                    N += 1
                    sum += 1

        else:
            for bi in range(bs):
                index = next_int()
                v = data[index]
                d = v.mri.reshape([61, 73, 61, 1])
                for axis in 0, 1, 2:
                    if rn(0, 1):
                        d = np.flip(d, axis)
                voxs[bi] = d
                # ox, oy, oz = rn(0, 2), rn(0, 2), rn(0, 2)
                # voxs[bi, ox:30 + ox, oy:30 + oy, oz:30 + oz] = d
                one_hots[bi][v.label] = 1
        if self._varbass:
            print(one_hots)
        return voxs, one_hots

    def data_iter(self,mode):
        while 1:
            for i in range(len(self._data[mode])):
                yield  i

    def get_fmri(self,mode):
        generator = iter(self.data_iter(mode)).__next__
        data = self._data[mode]
        while 1:
            index = generator()
            fmri = np.load(data[index]._fi)
            label = data[index].label
            yield fmri,label,data[index]._fi

    #fmri经过voxnet获取时间序列
    def get_time_batch(self,cut_shape,time_dim=40,batch_size=None,_batch_mode=None,_mode = None,flag = 0):
        # voxnet模型设置
        xyz = 32
        # p = dict()
        # p['output'] = voxnet['fc4']
        # p['output'] = tf.reshape(p['output'], [-1, 50])
        true_shape = []
        for x in range(0, len(cut_shape), 2):
            true_shape.append(cut_shape[x + 1] - cut_shape[x] + 1)
        # shape.append(1)
        if _batch_mode == None:
            _batch_mode = self._batch_mode
        if _mode == None:
            _mode = self._mode
        if flag == 1:
            _iter = iter(self.data_iter(_mode)).__next__
        while 1:
            self._batch_mode = _batch_mode
            self._mode = _mode
            # print(_batch_mode, _mode, self._batch_mode, self._mode)
            time_serial = np.ones([batch_size,time_dim,100], dtype=np.float32)
            time_serial = time_serial * -1
            one_hots = np.zeros([batch_size, self.num_categories], dtype=np.float32)
            data = self._data[_mode]
            total = 0

            # 获取两种不同的样本
            if _batch_mode == 'oversampling':
                # 当取样模式为过取样时,batch_size表示每一个样本取样的个数
                # print('\n',_batch_mode,_mode)
                for i in self._data_type:
                    for j in range(batch_size // 2):
                        index = self._iters[_batch_mode][_mode][i].__next__()
                        # 加载图片
                        # img = np.load(data[index]._fi)
                        # img = img.get_fdata()
                        # img = np.transpose(img, [3, 0, 1, 2])
                        # if self._varbass:
                        #     print(img.shape)
                        #
                        # new_feature = img[0:min(time_dim, img.shape[0]), cut_shape[0]:cut_shape[1] + 1,
                        #               cut_shape[2]:cut_shape[3] + 1,
                        #               cut_shape[4]:cut_shape[5] + 1]
                        # # 调整形状
                        # new_shape = [new_feature.shape[0],true_shape[0], true_shape[1],true_shape[2], 1]
                        # new_feature = np.reshape(new_feature, new_shape)
                        # start_t = (time_dim - new_feature.shape[0]) // 2
                        # start_x = (xyz - true_shape[0]) // 2
                        # start_y = (xyz - true_shape[1]) // 2
                        # start_z = (xyz - true_shape[2]) // 2
                        # voxs = np.zeros([new_feature.shape[0],xyz,xyz,xyz,1],np.float32)
                        # voxs[0:new_feature.shape[0],start_x:start_x + true_shape[0], start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = new_feature
                        # # 调整形状,形状为[80,mri维度，1]
                        # # new_shape = [min(time_dim, img.shape[0]), shape[1], shape[2], shape[3], 1]
                        # # new_feature = np.reshape(new_feature, new_shape)
                        # xx = 0
                        # time_feature = model.predict_on_batch(voxs)
                            # print(tf.get_default_graph())
                        # time_feature = sess.run(p['output'], feed_dict={voxnet[0]: voxs,voxnet.keep_prob:1.0,voxnet.training:False})
                        new_feature = data[index]._fi
                        new_feature = new_feature[0:min(time_dim,new_feature.shape[0])]
                        start_t = (time_dim - new_feature.shape[0]) // 2
                        time_serial[total][start_t:start_t+new_feature.shape[0]]=new_feature
                        one_hots[total, data[index].label] = 1
                        total = total + 1
                        if self._varbass:
                            print(total)
            if _batch_mode == 'random':
                for bs in range(batch_size):
                    # if flag == 0:
                    index = self._iters[_batch_mode][_mode].__next__()
                    if flag == 1:
                        index = _iter()
                    # print(index)
                    # print('\n',_batch_mode,_mode)
                    # 加载图片
                    # img = np.load(data[index]._fi)
                    # # img = img.get_fdata()
                    # # img = np.transpose(img, [3, 0, 1, 2])
                    # if self._varbass:
                    #     print(img.shape)
                    # new_feature = img[0:min(time_dim, img.shape[0]), cut_shape[0]:cut_shape[1] + 1,
                    #               cut_shape[2]:cut_shape[3] + 1,
                    #               cut_shape[4]:cut_shape[5] + 1]
                    # # 调整形状
                    # new_shape = [new_feature.shape[0], true_shape[0], true_shape[1], true_shape[2], 1]
                    # new_feature = np.reshape(new_feature, new_shape)
                    # #计算图像的坐标开始位置
                    # start_t = (time_dim - new_feature.shape[0]) // 2
                    # start_x = (xyz - true_shape[0]) // 2
                    # start_y = (xyz - true_shape[1]) // 2
                    # start_z = (xyz - true_shape[2]) // 2
                    # voxs = np.zeros([new_feature.shape[0], xyz, xyz, xyz,1], np.float32)
                    # voxs[0:new_feature.shape[0], start_x:start_x + true_shape[0],
                    # start_y:start_y + true_shape[1], start_z:start_z + true_shape[2], 0:1] = new_feature
                    # 调整形状,形状为[80,mri维度，1]
                    # new_shape = [min(time_dim, img.shape[0]), shape[1], shape[2], shape[3], 1]
                    # new_feature = np.reshape(new_feature, new_shape)
                    # time_feature = sess.run(p['output'], feed_dict={voxnet[0]: voxs,voxnet.keep_prob:1.0,voxnet.training:False})
                    # with Graph[0].as_default():
                    new_feature = data[index]._fi
                    new_feature = new_feature[0:min(time_dim, new_feature.shape[0])]
                    start_t = (time_dim - new_feature.shape[0]) // 2
                    time_serial[total][start_t:start_t + new_feature.shape[0]] = new_feature
                    one_hots[total, data[index].label] = 1
                    total = total + 1
                    if self._varbass:
                        print(total)
            # print(one_hots[0:total])
            yield time_serial, one_hots

    #获取时间片
    def get_time_stamp(self,time_len,time_dim):
        time_stamp = np.random.choice(np.arange(time_len), time_dim, replace=True)
        time_stamp.sort()
        return time_stamp

    #获取246数据
    def get_246_batch(self,mask,batch_size = None,time_dim=40,feature_index = []):
        #选取特征个数
        if feature_index == []:
            feature_len = 246
            index = np.linspace(1,246,246)
            feature_index = list(map(lambda x:int(x),index))
        else:
            feature_len = len(feature_index)

        #模板，时间维度
        bs = batch_size if batch_size is not None else self._batch_size
        bs = bs if bs is not None else 8
        time_serial = np.zeros([bs, time_dim, feature_len], dtype=np.float32)
        one_hots = np.zeros([bs, self.num_categories], dtype=np.float32)
        data = self._data[self._mode]
        next_int = self._iters[self._mode].__next__
        for bi in range(bs):
            index = next_int()
            v = data[index]
            # 加载图片
            img_data = np.load(v._fi)
            img_data = img_data.astype(np.float32)
            # 时间点选择
            time_stamp = np.linspace(0, img_data.shape[0] - 1, time_dim)
            time_stamp = list(map(lambda x: int(x), time_stamp))
            time_stamp = img_data[time_stamp]
            time_serial[bi] = time_stamp[:,feature_index].copy()
            one_hots[bi][v.label] = 1
            # print(time_serial[bi])
        return time_serial, one_hots

    #获取脑区原始体素
    def get_brain_batch(self,mask,batch_size = None,time_dim = 40,feature_index = []):
        bs = batch_size
        #特征长度
        #每一个特征的长度的前缀和
        One_len = [0]
        for i in feature_index:
            One_len.append(One_len[-1] + len(np.where(mask == i)[0]))
        time_serial = np.zeros([bs, time_dim, 25],dtype=np.float32)
        one_hots = np.zeros([bs, self.num_categories], dtype=np.float32)
        data = self._data[self._mode]
        bi = 0
        if self._batch_mode == 'oversampling':
            # 当取样模式为过取样时,batch_size表示每一个样本取样的个数
            for i in self._data_type:
                for j in range(batch_size // 2):
                    index = self._iters[self._batch_mode][self._mode][i].__next__()
                    # 加载图片
                    img_data = np.load(data[index]._fi)
                    # img = img.get_fdata()
                    # img = np.transpose(img, [3, 0, 1, 2])
                    # 时间点选择
                    time_stamp = img_data[0:min(img_data.shape[0],time_dim)]
                    # print(time_stamp.shape)
                    start_t = (time_dim - time_stamp.shape[0]) // 2
                    # 构造特征矩阵
                    # for f_index, feature in enumerate(feature_index):
                    #     tmp = np.where(mask == feature)
                    #     new_feature = time_stamp[:, tmp[0], tmp[1], tmp[2]]
                    time_serial[bi][start_t:start_t+time_stamp.shape[0]] = time_stamp[:,25:50]
                    one_hots[bi][data[index].label] = 1
                    bi += 1

        if self._batch_mode == 'random':
            for bs in range(batch_size):
                index = self._iters[self._batch_mode][self._mode].__next__()
                # 加载图片
                img_data = np.load(data[index]._fi)
                # img = img.get_fdata()
                # img = np.transpose(img, [3, 0, 1, 2])
                # 时间点选择
                time_stamp = img_data[0:min(img_data.shape[0], time_dim)]
                start_t = (time_dim - time_stamp.shape[0]) // 2
                # 构造特征矩阵
                time_serial[bi][start_t:start_t + time_stamp.shape[0]] = time_stamp[:,25:50]
                one_hots[bi][data[index].label] = 1
                bi += 1
        return time_serial, one_hots

    #获取截断后的sMRI
    def get_small_brain(self,cut_shape,batch_size):
        shape=[400]
        for x in range(0,len(cut_shape),2):
            shape.append(cut_shape[x + 1] - cut_shape[x] + 1)
        shape.append(1)
        voxs = np.zeros(shape,dtype=np.float32)
        one_hots = np.zeros([400,self.num_categories],dtype=np.float32)
        data = self._data[self._mode]
        total = 0
        #获取两个不同的样本
        if self._batch_mode == 'oversampling':
        #当取样模式为过取样时,batch_size表示每一个样本取样的个数
            for i in self._data_type:
                for j in range(batch_size//2):
                    index = self._iters[self._batch_mode][self._mode][i].__next__()
                    # 加载图片
                    img = nib.load(data[index]._fi)
                    img = img.get_fdata()
                    img = np.transpose(img, [3, 0, 1, 2])
                    if self._varbass:
                        print(img.shape)
                    new_feature = img[:, cut_shape[0]:cut_shape[1] + 1, cut_shape[2]:cut_shape[3] + 1,
                                  cut_shape[4]:cut_shape[5] + 1]
                    # 调整形状
                    new_shape = [img.shape[0], shape[1], shape[2], shape[3], 1]
                    new_feature = np.reshape(new_feature, new_shape)
                    voxs[total:total + img.shape[0]] = new_feature
                    one_hots[total:total + img.shape[0], data[index].label] = 1
                    total = total + img.shape[0]
                    if self._varbass:
                        print(total)
        if self._batch_mode == 'random':
            for bs in range(batch_size):
                index = self._iters[self._batch_mode][self._mode].__next__()
                # 加载图片
                img = nib.load(data[index]._fi)
                img = img.get_fdata()
                img = np.transpose(img, [3, 0, 1, 2])
                if self._varbass:
                    print(img.shape)
                new_feature = img[:, cut_shape[0]:cut_shape[1] + 1, cut_shape[2]:cut_shape[3] + 1,
                              cut_shape[4]:cut_shape[5] + 1]
                # 调整形状
                new_shape = [img.shape[0], shape[1], shape[2], shape[3], 1]
                new_feature = np.reshape(new_feature, new_shape)
                voxs[total:total + img.shape[0]] = new_feature
                one_hots[total:total + img.shape[0], data[index].label] = 1
                total = total + img.shape[0]
                if self._varbass:
                    print(total)
        # print(one_hots[0:total])
        return voxs[0:total],one_hots[0:total]



def main(_):
    start = time.time()
    dataset = fMRI_data(['MCIc', 'MCInc'], data_value=[0.7,0.3],varbass=True, dir="D:/fmri/fmri_data")
    img = nib.load('BN_Atlas_246_3mm.nii')
    mask = img.get_fdata()
    time_serial,one_hots = dataset.train.get_brain_batch(mask,batch_size=1,time_dim=40,feature_index=[1])
    print(one_hots)
    end = time.time()
    print((end-start)/60)
    #debug
    # cnt = [0 for i in range(246)]:
    # for i in range(246):
    #     for j in range(mask.shape[0]):
    #         for k in range(mask.shape[1]):
    #             for l in range(mask.shape[2]):
    #                 if mask[j][k][l] == i+1:
    #                     cnt[i] += 1
    #     print(cnt[i])
    # print(cnt)

if __name__ == '__main__':
   tf.app.run()
