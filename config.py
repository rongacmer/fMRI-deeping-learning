import tensorflow as tf

flag = tf.app.flags
cfg = flag.FLAGS

flag.DEFINE_boolean('varbass',False,'varbass') #调试


flag.DEFINE_string('output','../logs/MCIcvsMCInc_logs_4_2_1.txt','ouput filename') #日志文件
flag.DEFINE_boolean('istraining',True,'loading variables?')
flag.DEFINE_integer('checkpoint_start_num',0,'recording checkpoint num')
flag.DEFINE_string('voxnet_checkpoint','/home/anzeng/rhb/fmri/ADvsNC_fcn_checkpoint_4_9/voxnet-5.npz','voxnet checkpoint') #voxnet检查点
flag.DEFINE_string('fcn_checkpoint','/home/anzeng/rhb/fmri/ADvsNC_fcn_checkpoint_4_9/fcn-5.npz','fcn_checkpoint')

flag.DEFINE_string('voxnet_checkpoint_dir','/home/anzeng/rhb/fmri/ADvsNC_voxnent_checkpoint_4_9','checkpoint_dir') #保存的检查点路径
flag.DEFINE_string('fcn_checkpoint_dir','/home/anzeng/rhb/fmri/ADvsNC_fcn_checkpoint_4_9','checkpoint_dir') #保存的检查点路径

print(cfg.varbass)