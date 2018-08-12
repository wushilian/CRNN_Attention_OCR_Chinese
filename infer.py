from model import *
import config as cfg
import time
import os

loss,train_decode_result,pred_decode_result=build_network(is_training=False)
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list,max_to_keep=5)

sess = tf.Session()

ckpt = tf.train.latest_checkpoint(cfg.CKPT_DIR)
if ckpt:
    saver.restore(sess,ckpt)
    print('restore from the checkpoint{0}'.format(ckpt))
else:
    print('failed to load ckpt')
val_img,_=cfg.read_data(cfg.val_dir)
val_predict = sess.run(pred_decode_result,feed_dict={image: val_img})
predit = cfg.int2label(np.argmax(val_predict, axis=2))
print(predit)
