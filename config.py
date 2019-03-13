import tensorflow as tf

flag = tf.app.flags
cfg = flag.FLAGS

flag.DEFINE_string('output','logs/ADvsNC_logs_3_12_1.txt','ouput filename') #日志文件
flag.DEFINE_string('voxnet_checkpoint_dir','/home/anzeng/rhb/fmri/fMRI-deeping-learning/checkpoints/c-157.npz','voxnet checkpoint') #voxnet检查点
flag.DEFINE_string('fcn_checkpoint_dir','/home/anzeng/rhb/fmri/fMRI-deeping-learning/checkpoing_fcns/c-100.npz','fcn_checkpoint')
