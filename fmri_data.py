import nibabel as nib
import os
import random
import numpy as np
import tensorflow as tf
from config import cfg
from voxnet import VoxNet
# filename = 'D:/fmri/raw_AD/GretnaFunNIfTI/006_S_4153/xbcNGSdswranrest.nii'
# img = nib.load(filename)
# img_data = img.get_fdata()
# img_data = np.transpose(img_data,[3,0,1,2])
# print(img_data.shape)

class fMRI_data(object):

    def __init__(self, data_type=['AD','NC'],batch_size=None,varbass = False,dir="/home/anzeng/rhb/fmri_data"):
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
        self._varbass = varbass
        self._mode = 'train'
        self._data_type=data_type
        self._dir = dir  #地址索引
        self._iters = {}
        self._data = {'train': [], 'test': []}
        ###################################################
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

        for k in self._data:
            self._data[k] = [MRI(i, categories[c], c) for c, i in self._data[k]]
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
    def get_time_batch(self,time_dim=40,batch_size=None):
        #voxnet模型设置
        voxnet=VoxNet()
        p=dict()
        p['output'] = voxnet['fc_1']
        p['output'] = tf.reshape(p['output'],[None,128])
        rn = random.randint
        #批次大小
        bs = batch_size if batch_size is not None else self._batch_size
        bs = bs if bs is not None else 4
        time_serial = np.zeros([bs,time_dim,128],dtype=np.float32)
        one_hots = np.zeros([bs,self.num_categories],dtype=np.float32)
        data = self._data[self._mode]
        next_int = self._iters[self._mode].__next__
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(voxnet.npz_saver.restore(sess,cfg.voxnet_checkpoint_dir))
            for bi in range(bs):
                index = next_int()
                v = data[index]
                #加载图片
                img = nib.load(v._fi)
                img_data = img.get_fdata()
                img_data = np.transpose(img_data, [3, 0, 1, 2])
                #时间点选择
                time_stamp = np.linspace(0,img_data.shape[0],time_dim)
                time_stamp = list(map(lambda x:int(x),time_stamp))
                time_stamp_img = img_data[time_stamp]
                #获取特征
                time_feature = sess.run(p['output'],feed_dict={voxnet[0]:time_stamp_img})
                time_serial[bi] = time_feature
                one_hots[bi][v.label] = 1
                if self._varbass:
                    print(time_feature.shape)
        return time_serial,one_hots


if __name__ == '__main__':
    data = fMRI_data(['AD','NC'],varbass=True)
    voxs,one_hots =data.get_batch(5)
    print(voxs.shape)
