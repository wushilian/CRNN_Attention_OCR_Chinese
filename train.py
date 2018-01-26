
from model import *
import config as cfg
import time
import os
from sklearn.utils import shuffle
loss,train_decode_result,pred_decode_result=build_network(is_training=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.learning_rate, momentum=cfg.momentum, use_nesterov=True)
train_op=optimizer.minimize(loss)
saver = tf.train.Saver(max_to_keep=5)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
with tf.name_scope('summaries'):
    tf.summary.scalar("cost", loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(cfg.LOGS_PATH)


if cfg.is_restore:
    ckpt = tf.train.latest_checkpoint(cfg.CKPT_DIR)
    if ckpt:
        saver.restore(sess,ckpt)
        print('restore from the checkpoint{0}'.format(ckpt))
img,label=cfg.read_data('../dataset/Chinese/','train.txt')
#img,label=cfg.read_data('test','test.txt')
val_img,val_label=cfg.read_data('test','test.txt')
num_train_samples=img.shape[0]
num_batches_per_epoch = int(num_train_samples/cfg.BATCH_SIZE)
target_in,target_out=cfg.label2int(label)
for cur_epoch in range(cfg.EPOCH):

    shuffle_idx = np.random.permutation(num_train_samples)
    train_cost = 0
    start_time = time.time()
    batch_time = time.time()
    # the tracing part
    for cur_batch in range(num_batches_per_epoch):
        val_img,val_label=shuffle(val_img,val_label)
        batch_time = time.time()
        indexs = [shuffle_idx[i % num_train_samples] for i in
                  range(cur_batch * cfg.BATCH_SIZE, (cur_batch + 1) * cfg.BATCH_SIZE)]
        batch_inputs,batch_target_in,batch_target_out=img[indexs],target_in[indexs],target_out[indexs]
        sess.run( train_op,feed_dict={image: batch_inputs,train_output: batch_target_in,target_output: batch_target_out,sample_rate:np.min([1.,0.2*cur_epoch+0.2])})
        if cur_batch%cfg.DISPLAY_STEPS==0:
            summary_loss, loss_result = sess.run([summary_op, loss],feed_dict={image: batch_inputs,train_output: batch_target_in,target_output: batch_target_out,
                                                                               sample_rate: np.min([1., 1.])})
            writer.add_summary(summary_loss, cur_epoch*num_batches_per_epoch+cur_batch)
            val_predict = sess.run(pred_decode_result,feed_dict={image: val_img[0:cfg.BATCH_SIZE]})
            train_predict = sess.run(pred_decode_result, feed_dict={image: batch_inputs, train_output: batch_target_in,
                                                                     target_output: batch_target_out,sample_rate:np.min([1., 1.])})
            predit = cfg.int2label(np.argmax(val_predict, axis=2))
            train_pre = cfg.int2label(np.argmax(train_predict, axis=2))
            gt = val_label[0:cfg.BATCH_SIZE]
            acc = cfg.cal_acc(predit, gt)
            print("epoch:{}, batch:{}, loss:{}, acc:{},\n train_decode:{}, \n val_decode:{}, \n ground_truth:{}".
                  format(cur_epoch, cur_batch,
                         loss_result, acc,
                         train_pre[0:5],
                         predit[0:5],
                         gt[0:5]))
            if not os.path.exists(cfg.CKPT_DIR):
                os.makedirs(cfg.CKPT_DIR)
            saver.save(sess, os.path.join(cfg.CKPT_DIR, 'attention_ocr.model'), global_step=cur_epoch*num_batches_per_epoch+cur_batch)
