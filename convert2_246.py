import numpy as np
import nibabel as nib
import os

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
#参数设置
pre_dir ='/home/anzeng/rhb/fmri_data/246_feature_data'
pre_raw_dir = '/home/anzeng/rhb/fmri_data'
mode = 'AD'
mask = ''
data_mode=['train','test']
mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
mask = mask.get_fdata()
count = 0
train_dir = os.path.join(pre_dir,mode,data_mode[0])
test_dir = os.path.join(pre_dir,mode,data_mode[1])
pre_raw_dir = os.path.join(pre_raw_dir,mode)
pre_dir = os.path.join(pre_dir,mode)
if not os.path.isdir(train_dir):
    mkdir(train_dir)
if not os.path.isdir(test_dir):
    mkdir(test_dir)

for i in data_mode:
    #原始数据路径
    now_dir = os.path.join(pre_raw_dir,i)
    #处理后数据路径
    MRI_dir = os.path.join(pre_dir,i)
    for lists in os.listdir(now_dir):
        sub_path = os.path.join(now_dir, lists)
        img = nib.load(sub_path)
        img_data = img.get_fdata()
        img_data = np.transpose(img_data, [3, 0, 1, 2])
        features_data = np.zeros([img_data.shape[0],246])
        for cnt in range(246):
            now_mask = mask.copy()
            now_mask[np.where(now_mask != cnt + 1)] = 0
            now_mask[np.where(now_mask == cnt + 1)] = 1
            sum = np.sum(now_mask)
            for index in range(img_data.shape[0]):
                voxs = np.sum(np.multiply(img_data[index], now_mask)) / sum
                features_data[index][cnt] = voxs
        filename = os.path.join(MRI_dir,mode+'_'+str(count))
        np.save(filename,features_data)
        count += 1
    print(i+" convert success")
