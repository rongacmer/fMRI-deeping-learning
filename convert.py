import numpy as np
import nibabel as nib
import os
import shutil
def mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def normalization():
    brain_map = [215, 216, 217, 218]
    BN_mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
    BN_mask = BN_mask.get_fdata()
    mask = np.zeros(BN_mask.shape, np.float32)
    for i in brain_map:
        mask[np.where(BN_mask == i)] = 1
    dir = '/home/anzeng/rhb/fmri_data/new_fmri'
    mode = 'AD'
    raw_dir = os.path.join(dir,'raw_'+mode)
    target_dir = os.path.join(dir,mode)
    mkdir(target_dir)
    for i in os.listdir(raw_dir):
        if i[0] == 'x':
            continue
        filename = os.path.join(raw_dir,i)
        img = nib.load(filename)
        img = img.get_fdata()
        img = np.transpose(img,[3,0,1,2])
        new_data = np.zeros(img.shape, dtype=np.float32)
        for index in range(img.shape[0]):
            #归一化数据
            new_data[index] = (img[index] - np.mean(img[index])) / np.std(img[index])
            new_data[index] = np.multiply(new_data[index],mask)
        filename = os.path.join(target_dir,i.split('.')[0])
        np.save(filename,new_data)
        print(i + ' convert success')

def make_new_data():
    #参数设置
    feature_index=[217]
    mask_dir = '/home/anzeng/rhb/fmri_data/217'
    pre_dir ='/home/anzeng/rhb/fmri_data/MRI_data/217'
    pre_raw_dir = '/home/anzeng/rhb/fmri_data'
    mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
    mask = mask.get_fdata()
    mode = 'MCIc'
    data_mode=['train','test']
    count = 0
    #原始数据路径
    pre_raw_dir = os.path.join(pre_raw_dir,mode)
    #sMRI路径
    pre_dir = os.path.join(pre_dir,mode)
    #mask路径
    mask_dir = os.path.join(mask_dir,mode)

    for i in data_mode:
        #原始数据路径
        now_dir = os.path.join(pre_raw_dir,i)
        #处理后数据路径
        MRI_dir = os.path.join(pre_dir,i)
        mask_now_dir = os.path.join(mask_dir,i)
        mkdir(MRI_dir)
        mkdir(mask_now_dir)
        for lists in os.listdir(now_dir):
            sub_path = os.path.join(now_dir, lists)
            img = nib.load(sub_path)
            img_data = img.get_fdata()
            img_data = np.transpose(img_data, [3, 0, 1, 2])
            new_data = np.zeros(img_data.shape,dtype=np.float32)
            for cnt in range(img_data.shape[0]):
                # 路径地址
                filename = os.path.join(MRI_dir, mode + '_'+str(count))
                for index in feature_index:
                    new_data[cnt][np.where(mask==index)] = img_data[cnt][np.where(mask==index)]
                data = new_data[cnt].astype(np.float32)
                np.save(filename, data)
                count += 1
            filename = os.path.join(mask_now_dir,lists)
            np.save(filename,new_data)
            print(new_data.shape)
        print(i+" convert success")

#转换为时间矩阵
def convert2time(raw_dir,target_dir,train_list,test_list,brain_map,BN_mask):
    One_len = [0]
    for i in brain_map:
        One_len.append(One_len[-1] + len(np.where(BN_mask == i)[0]))
    p = dict()
    p['train_dir'] = os.path.join(target_dir, 'train')
    p['test_dir'] = os.path.join(target_dir, 'test')
    p['train_list'] = train_list
    p['test_list'] = test_list
    mkdir(p['train_dir'])
    mkdir(p['test_dir'])
    mode = ['train', 'test']
    filename_list = os.listdir(raw_dir)
    for x in mode:
        for i in p[x + '_list']:
            sub_path = os.path.join(raw_dir, filename_list[i])
            img_data = np.load(sub_path)
            data = np.zeros([img_data.shape[0],One_len[-1]])
            # img_data = img.get_fdata()
            # img_data = np.transpose(img_data, [3, 0, 1, 2])
            # new_data = np.zeros(img_data.shape, dtype=np.float32)
            for f_index, feature in enumerate(brain_map):
                tmp = np.where(BN_mask == feature)
                new_feature = img_data[:, tmp[0], tmp[1], tmp[2]]
                data[:,One_len[f_index]:One_len[f_index + 1]] = new_feature
                # data = img_data[cnt]
                # data = np.multiply(data, mask)
                # 图片归一化
                # diff = np.max(data) - np.min(data)
                # print('diff:',diff)
                # if diff < esp:
                #     continue
                # data = (data - np.min(data)) / diff
            filename = os.path.join(p[x + '_dir'], filename_list[i].split('.')[0])
            np.save(filename, data)
            # filename = os.path.join(mask_now_d

def covert2smri(raw_dir,target_dir,train_list,test_list,brain_map,BN_mask):
    # mask = np.zeros(BN_mask.shape,np.float32)
    # for i in brain_map:
    #     mask[np.where(BN_mask == i)] = 1
    # print(np.count_nonzero(mask))
    esp = 1e-5
    p = dict()
    p['train_dir'] = os.path.join(target_dir,'train')
    p['test_dir'] = os.path.join(target_dir,'test')
    p['train_list'] = train_list
    p['test_list'] = test_list
    mkdir(p['train_dir'])
    mkdir(p['test_dir'])
    mode = ['train','test']
    filename_list = os.listdir(raw_dir)
    for x in mode:
        for i in p[x+'_list']:
            sub_path = os.path.join(raw_dir, filename_list[i])
            img_data = np.load(sub_path)
            # img_data = img.get_fdata()
            # img_data = np.transpose(img_data, [3, 0, 1, 2])
            # new_data = np.zeros(img_data.shape, dtype=np.float32)
            for cnt in range(img_data.shape[0]):
                # 路径地址
                filename = os.path.join(p[x+'_dir'], filename_list[i].split('.')[0] + '_' + str(cnt))
                # for index in feature_index:
                #     new_data[cnt][np.where(mask == index)] = img_data[cnt][np.where(mask == index)]
                data = img_data[cnt]
                # data = np.multiply(data,mask)
                #图片归一化
                # diff = np.max(data) - np.min(data)
                # print('diff:',diff)
                # if diff < esp:
                #     continue
                # data = (data - np.min(data)) / diff
                np.save(filename, data)
            # filename = os.path.join(mask_now_dir, lists)
            # np.save(filename, new_data)
            # print(new_data.shape)

# normalization()
# brain_map = [109,110,211,212,215,216,217,218]
brain_map = [218]
cut_shape = [100, 0, 100, 0, 100, 0]
mask = nib.load('BN_Atlas_246_3mm.nii')
mask = mask.get_fdata()
# 获取截取的sMRI大小
for x in brain_map:
    tmp = np.where(mask == x)
    for i in range(3):
        cut_shape[2 * i] = min(cut_shape[2 * i], np.min(tmp[i]))
        cut_shape[2 * i + 1] = max(cut_shape[2 * i + 1], np.max(tmp[i]))
print(brain_map, cut_shape)
s = set()
for i in range(cut_shape[0],cut_shape[1]+1,1):
    for j in range(cut_shape[2],cut_shape[3]+1):
        for k in range(cut_shape[4],cut_shape[5]+1):
            s.add(mask[i,j,k])
print(s)
BN_mask = nib.load('BN_Atlas_246_3mm.nii')
BN_mask = BN_mask.get_fdata()
mask = np.zeros(BN_mask.shape,np.float32)
for i in brain_map:
    mask[np.where(BN_mask == i)] = 1
print(mask.shape)
# img = nib.load('sub-OAS30078.nii')
# img_data = img.get_fdata()
# img_data = np.transpose(img_data, [3, 0, 1, 2])
# new_data = np.zeros(img_data.shape, dtype=np.float32)
# print(np.max(new_data) - np.min(new_data))
# make_new_data()
# normalization()