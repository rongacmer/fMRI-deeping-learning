import numpy as np
import nibabel as nib
import os

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

train_dir ='D:/fmri/MRI_data/AD/train'
test_dir = 'D:/fmri/MRI_data/AD/test'
raw_dir = 'D:/fmri/fmri_data/AD'
train_size = 0.8
mode = 'AD_'
count = 0
mkdir(train_dir)
mkdir(test_dir)
dir_list = os.listdir(raw_dir)
for j,lists in enumerate(dir_list):
    sub_path = os.path.join(raw_dir, lists)
    img = nib.load(sub_path)
    img_data = img.get_fdata()
    img_data = np.transpose(img_data, [3, 0, 1, 2])
    for cnt in range(img_data.shape[0]):
        #路径地址
        MRI_dir = train_dir if j < len(dir_list) * train_size else test_dir
        filename = os.path.join(MRI_dir,mode+str(count))
        data = img_data[cnt].astype(np.float32)
        np.save(filename,data)
        count += 1
