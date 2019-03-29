import numpy as np
import nibabel as nib
import os

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_smri():
    #参数设置
    pre_dir ='/home/anzeng/rhb/fmri_data/MRI_data'
    pre_raw_dir = '/home/anzeng/rhb/fmri_data'
    mask = nib.load('/home/anzeng/rhb/fmri/fMRI-deeping-learning/BN_Atlas_246_3mm.nii')
    mask = mask.get_fdata()
    mode = 'EMCI'
    data_mode=['train','test']
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
            for cnt in range(img_data.shape[0]):
                # 路径地址
                filename = os.path.join(MRI_dir, mode + '_'+str(count))
                data = img_data[cnt].astype(np.float32)
                np.save(filename, data)
                count += 1
        print(i+" convert success")
