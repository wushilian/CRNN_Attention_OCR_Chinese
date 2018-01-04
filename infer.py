from model import *
import config as cfg
import time
import os

loss,train_decode_result,pred_decode_result=build_network(is_training=True)
saver = tf.train.Saver()

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
