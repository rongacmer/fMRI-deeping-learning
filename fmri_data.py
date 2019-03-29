import nibabel as nib
import os
import random
import numpy as np
import tensorflow as tf
import time

from config import cfg
from voxnet import VoxNet
# filename = 'D:/fmri/raw_AD/GretnaFunNIfTI/006_S_4153/xbcNGSdswranrest.nii'
# img = nib.load(filename)
# img_data = img.get_fdata()
# img_data = np.transpose(img_data,[3,0,1,2])
# print(img_data.shape)

class fMRI_data(object):

    #不同类别赋予不同的权值
    def __init__(self, data_type=['AD','NC'],data_value=[0.5,0.5],batch_size=None,varbass = False,dir="/home/anzeng/rhb/fmri_data"):
        class MRI(object):
            def __init__(self, fi, label, category,value):
                #fi:路径名,label:标签,category:类别
                self._fi = fi
                self._label = label
                self._category = category
                self._value = value

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
            def value(self):
                return self._value

            @property
            def filename(self):
                return self._fi.filename.split('/')[-1]

            def save(self, f=None):
                self.filename if f is None else f
                np.save(f, self.mri)

        ############初始化数据集信息######################
        self._batch_size = batch_size
        self._varbass = varbass
        self._mode = 'train'
        self._data_type=data_type
        self._dir = dir  #地址索引
        self._iters = {}
        self._data = {'train': [], 'test': []}
        ###################################################

        ########迭代器############
        def get_random_iter(mode):
            while 1:
                order = np.arange(len(self._data[mode]))
                np.random.shuffle(order)
                for i in order:
                    yield i
        print('Setting up ' +str(self._data_type)+'database... ')

        #加载数据，模型:标签，文件路径
        for classification in self._data_type:
            now_dir = os.path.join(self._dir,classification)
            for mode in ['train','test']:
                now_dir_x = os.path.join(now_dir,mode)
                for i in os.listdir(now_dir_x):
                    category = classification
                    train_or_test=mode
                    filename = os.path.join(now_dir_x,i)
                    self._data[train_or_test].append((category,filename))

        #标签制作
        categories = sorted(list(set(c for c, i in self._data['test'])))
        categories = dict(zip(categories, range(len(categories))))
        categories_value = dict(zip(data_type,data_value))

        for k in self._data:
            self._data[k] = [MRI(i, categories[c], c,categories_value[c]) for c, i in self._data[k]]
            self._iters[k] = iter(get_random_iter(k))
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
    def data(self):
        return self._data[self._mode]

    def __len__(self):
        return len(self._data[self._mode])

    def get_batch(self, batch_size=None):
        rn = random.randint
        bs = batch_size if batch_size is not None else self._batch_size
        bs = bs if bs is not None else 16
        voxs = np.zeros([bs, 61, 73, 61, 1], dtype=np.float32)
        one_hots = np.zeros([bs, self.num_categories], dtype=np.float32)
        data = self._data[self._mode]
        next_int = self._iters[self._mode].__next__
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
        return voxs, one_hots

    #fmri经过voxnet获取时间序列
    def get_time_batch(self,sess,voxnet,time_dim=40,batch_size=None):
        # voxnet模型设置
        p=dict()
        p['output'] = voxnet['gap']
        p['output'] = tf.reshape(p['output'],[-1,128])
        # 批次大小
        bs = batch_size if batch_size is not None else self._batch_size
        bs = bs if bs is not None else 8
        time_serial = np.zeros([bs,time_dim,128],dtype=np.float32)
        one_hots = np.zeros([bs,self.num_categories],dtype=np.float32)
        data = self._data[self._mode]
        next_int = self._iters[self._mode].__next__
        for bi in range(bs):
            index = next_int()
            v = data[index]
            #加载图片
            img = nib.load(v._fi)
            img_data = img.get_fdata()
            img_data = np.transpose(img_data, [3, 0, 1, 2])
            img_data = np.reshape(img_data,[-1,61,73,61,1])
            img_data = img_data.astype(np.float32)
            #时间点选择
            time_stamp = np.linspace(0,img_data.shape[0]-1,time_dim)
            time_stamp = list(map(lambda x:int(x),time_stamp))
            time_stamp_img = img_data[time_stamp]
            #获取特征
            time_feature = sess.run(p['output'],feed_dict={voxnet[0]:time_stamp_img})
            if self._varbass:
                print(time_feature.shape)
            time_serial[bi] = time_feature
            one_hots[bi][v.label] = v.value
        return time_serial,one_hots

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
            one_hots[bi][v.label] = v.value
            # print(time_serial[bi])
        return time_serial, one_hots

    #获取脑区原始体素
    def get_brain_batch(self,mask,batch_size = None,time_dim = 40,feature_index = []):
        bs = batch_size if batch_size is not None else self._batch_size
        bs = bs if bs is not None else 8
        #特征长度
        #每一个特征的长度的前缀和
        One_len = [0]
        for i in feature_index:
            One_len.append(One_len[-1] + len(np.where(mask == i)[0]))

        time_serial = np.zeros([bs, time_dim, One_len[-1]],dtype=np.float32)
        one_hots = np.zeros([bs, self.num_categories], dtype=np.float32)
        data = self._data[self._mode]
        next_int = self._iters[self._mode].__next__
        for bi in range(bs):
            index = next_int()
            v = data[index]
            # 加载图片
            img_data = nib.load(v._fi)
            img_data = img_data.get_fdata()
            img_data = np.transpose(img_data,[3,0,1,2])
            #时间点选择
            time_stamp = img_data[self.get_time_stamp(img_data.shape[0],time_dim)]
            #构造特征矩阵
            for f_index,feature in enumerate(feature_index):
                tmp = np.where(mask == feature)
                new_feature = time_stamp[:,tmp[0],tmp[1],tmp[2]]
                time_serial[bi][:,One_len[f_index]:One_len[f_index+1]] = new_feature
            one_hots[bi][v.label] = v.value
        return time_serial, one_hots

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
