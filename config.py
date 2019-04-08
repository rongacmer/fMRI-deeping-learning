import tensorflow as tf

flag = tf.app.flags
cfg = flag.FLAGS

flag.DEFINE_boolean('varbass',False,'varbass') #调试

flag.DEFINE_string('output','logs/MCIcvsMCInc_logs_4_1_1.txt','ouput filename') #日志文件
flag.DEFINE_boolean('istraining',False,'loading variables?')
flag.DEFINE_string('voxnet_checkpoint','/home/anzeng/rhb/fmri/EMCIvsLMCI_checkpoint/cx-8,npz.npz','voxnet checkpoint') #voxnet检查点
flag.DEFINE_string('fcn_checkpoint','/home/anzeng/rhb/fmri/fMRI-deeping-learning/checkpoing_fcns/c-100.npz','fcn_checkpoint')

flag.DEFINE_string('voxnet_checkpoint_dir','/home/anzeng/rhb/fmri/EMCIvsLMCI_voxnent_checkpoint_LOO','checkpoint_dir') #保存的检查点路径
flag.DEFINE_string('fcn_checkpoint_dir','/home/anzeng/rhb/fmri/EMCIvsLMCI_fcn_checkpoint_LOO','checkpoint_dir') #保存的检查点路径