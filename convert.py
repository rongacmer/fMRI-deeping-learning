import numpy as np
import nibabel as nib
import os

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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

make_new_data()