from model import *
import config as cfg
import time
import os

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
img,label=cfg.read_data(cfg.train_dir)
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
        batch_time = time.time()
        indexs = [shuffle_idx[i % num_train_samples] for i in
                  range(cur_batch * cfg.BATCH_SIZE, (cur_batch + 1) * cfg.BATCH_SIZE)]
        batch_inputs,batch_target_in,batch_target_out=img[indexs],target_in[indexs],target_out[indexs]
        sess.run( train_op,feed_dict={image: batch_inputs,train_output: batch_target_in,target_output: batch_target_out})
        if cur_batch%cfg.DISPLAY_STEPS==0:
            summary_loss, loss_result = sess.run([summary_op, loss],feed_dict={image: batch_inputs,train_output: batch_target_in,target_output: batch_target_out})
            writer.add_summary(summary_loss, cur_epoch*num_batches_per_epoch+cur_batch)
            infer_predict = sess.run(pred_decode_result,feed_dict={image: batch_inputs,train_output: batch_target_in,target_output: batch_target_out})
            predit=cfg.int2label(np.argmax(infer_predict, axis=2))
            gt=cfg.int2label(batch_target_out)
            acc=cfg.cal_acc(predit,gt)
            print("epoch:{}, batch:{}, loss:{}, acc:{},\n predict_decode:{}, \n ground_truth:{}".
                  format(cur_epoch,cur_batch,
                         loss_result,acc,
                         predit[0:10],
                         gt[0:10]))
            if not os.path.exists(cfg.CKPT_DIR):
                os.makedirs(cfg.CKPT_DIR)
            saver.save(sess, os.path.join(cfg.CKPT_DIR, 'attention_ocr.model'), global_step=cur_epoch*num_batches_per_epoch+cur_batch)



