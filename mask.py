# from nilearn import masking
# mask = masking.compute_epi_mask(r'xbcNGSdswranrest(1).nii')
# print(mask.get_data().shape)
# from nilearn.masking import apply_mask
# masked_data = apply_mask(r'xbcNGSdswranrest(1).nii', mask)
# print(masked_data.shape)
# # masked_data shape is (timepoints, voxels). We can plot the first 150
# # timepoints from two voxels
#
# # And now plot a few of these
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 5))
# plt.plot(masked_data[:230, 98:100])
# plt.xlabel('Time [TRs]', fontsize=16)
# plt.ylabel('Intensity', fontsize=16)
# plt.xlim(0, 150)
# plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
#
# plt.show()

# import numpy as np
# for i in range(10):
#     x = np.random.choice([1,2,3],1)
#     print(x)
#
# a = np.zeros([5,5])
# a[0,0] = 1
# print(a)
# b = np.flip(a,axis=0)
# print(b)

# import tensorflow as tf
# import numpy as np
# x= [[0,1],[0,1],[1,0],[0,1],[0,1],[0,1]]
# w = [2,1,1]
# input_weight =  tf.placeholder(tf.float32,None)
# weight = tf.reshape(input_weight,[-1,1])
# sum_w = tf.reduce_sum(weight)
# input = tf.placeholder(tf.float32,[None,2])
# output = tf.argmax(input,axis=1)
# output = tf.cast(output,tf.float32)
# output = tf.reshape(output,[-1,3]) #python维度自动推导
# shape = tf.shape(output)
# ans = tf.matmul(output,weight)
# ans = tf.divide(ans,sum_w)
# ans = tf.round(ans)
#
# with tf.Session() as sess:
#     # x = sess.run(weight,feed_dict={weight:x})
#     t = sess.run(ans,feed_dict={input:x,input_weight:w})
#     print(t)
#


# import nibabel as nib
import matplotlib.pyplot as plt


# def read_data(path):
#     image_data = nib.load(path).get_data()
#     return image_data
#
#
# def show_img(ori_img):
#     plt.imshow(ori_img[:, :, 85], cmap='gray')  # channel_last
#     plt.show()
#
#
# path = 'F:/my_data/t1ce.nii.gz'
# data = read_data(path)
# show_img(data)

import nibabel as nib
import numpy as np
img = nib.load('sub-OAS30078.nii')
img = img.get_data()
img = np.transpose(img,[3,0,1,2])
BN_mask = nib.load('BN_Atlas_246_3mm.nii')
BN_mask = BN_mask.get_fdata()
mask = np.zeros(BN_mask.shape,np.float32)
# mask[np.where(BN_mask != 0)] = 1
# 获取截取的sMRI大小
brain_map = [110]
cut_shape=[100,0,100,0,100,0]
total_len = 0
# for x in brain_map:
#     tmp = np.where(BN_mask == x)
#     total_len += len(tmp[0])
#     mask[tmp] = 1
#     for i in range(3):
#         cut_shape[2 * i] = min(cut_shape[2 * i], np.min(tmp[i]))
#         cut_shape[2 * i + 1] = max(cut_shape[2 * i + 1], np.max(tmp[i]))
# print(brain_map, cut_shape)
tmp = np.where(BN_mask != 0)
total_len += len(tmp[0])
# mask[tmp] = 1
for i in range(3):
    cut_shape[2 * i] = min(cut_shape[2 * i], np.min(tmp[i]))
    cut_shape[2 * i + 1] = max(cut_shape[2 * i + 1], np.max(tmp[i]))
print(cut_shape)
# new_img = np.multiply(img[20],mask)
# print(new_img[cut_shape[0]:cut_shape[1] + 1, cut_shape[2]:cut_shape[3] + 1,
#                               cut_shape[4]:cut_shape[5] + 1])
# print(new_img)
# print(total_len)
# plt.imshow(img[0][25][:,:],cmap='gray')
# plt.show()
# print(np.max(img[0]) - np.min(img[0]))
# print(np.max(img[1]) - np.min(img[1])
import keras