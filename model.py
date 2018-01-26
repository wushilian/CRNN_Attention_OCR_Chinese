import tensorflow as tf
import config as cfg
from tensorflow.contrib import layers
from tensorflow.python.layers.core import Dense
import numpy as np
slim=tf.contrib.slim
image = tf.placeholder(tf.float32, shape=(None,cfg.IMAGE_WIDTH,cfg.IMAGE_HEIGHT, 1), name='img_data')
train_output = tf.placeholder(tf.int64, shape=[None, None], name='train_output')
target_output = tf.placeholder(tf.int64, shape=[None, None], name='target_output')
sample_rate=tf.placeholder(tf.float32, shape=[], name='sample_rate')
train_length=np.array([20]*cfg.BATCH_SIZE,dtype=np.int32)

def encoder_net(_image, scope,is_training,reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        net = tf.layers.batch_normalization(_image, training=is_training)
        net = slim.conv2d(net, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, 256, [3, 3], activation_fn=None, scope='conv3')
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, 256, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool3')
        net = slim.conv2d(net, 512, [3, 3], activation_fn=None, scope='conv5')
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, 512, [3, 3], scope='conv6')
        net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool4')
        net = slim.conv2d(net, 512, [2, 2], padding='VALID', activation_fn=None, scope='conv7')
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)#CRNN
        cnn_out = tf.squeeze(net,axis=2)

        cell = tf.contrib.rnn.GRUCell(num_units=cfg.RNN_UNITS)
        enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=cnn_out,dtype=tf.float32)#双向LSTM
        encoder_outputs = tf.concat(enc_outputs, -1)
        return encoder_outputs,enc_state

def decode(helper, memory, scope, enc_state,reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=cfg.RNN_UNITS, memory=memory)
        cell = tf.contrib.rnn.GRUCell(num_units=cfg.RNN_UNITS)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=cfg.RNN_UNITS, output_attention=True)
        output_layer = Dense(units=cfg.VOCAB_SIZE)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attn_cell, helper=helper,
            initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=cfg.BATCH_SIZE).clone(cell_state=enc_state[0]),
            output_layer=output_layer)
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=False,
            impute_finished=True, maximum_iterations=20)
        return outputs
def build_network(is_training):
    train_output_embed,enc_state= encoder_net(image, 'encode_features',is_training)

#vocab_size: 输入数据的总词汇量，指的是总共有多少类词汇，不是总个数，embed_dim：想要得到的嵌入矩阵的维度
    output_embed = layers.embed_sequence(train_output, vocab_size=cfg.VOCAB_SIZE, embed_dim=cfg.VOCAB_SIZE, scope='embed')#有种变为one-hot的意味
    embeddings = tf.Variable(tf.truncated_normal(shape=[cfg.VOCAB_SIZE, cfg.VOCAB_SIZE], stddev=0.1), name='decoder_embedding')#embdding变为类别

    start_tokens = tf.zeros([cfg.BATCH_SIZE], dtype=tf.int64)

    train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(output_embed, train_length,
                                                                       embeddings, sample_rate)

    #用于inference阶段的helper，将output输出后的logits使用argmax获得id再经过embedding layer来获取下一时刻的输入。
    #start_tokens： batch中每个序列起始输入的token_id  end_token：序列终止的token_id
    #start_tokens: int32 vector shaped [batch_size], the start tokens.
    #end_token: int32 scalar, the token that marks end of decoding.
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)#GO,EOS的序号
    train_outputs = decode(train_helper, train_output_embed,'decode',enc_state)

    pred_outputs = decode(pred_helper, train_output_embed, 'decode',enc_state, reuse=True)
    train_decode_result = train_outputs[0].rnn_output[:, :-1, :]
    pred_decode_result = pred_outputs[0].rnn_output
    mask = tf.cast(tf.sequence_mask(cfg.BATCH_SIZE * [train_length[0] - 1], train_length[0]),
                   tf.float32)
    att_loss = tf.contrib.seq2seq.sequence_loss(train_outputs[0].rnn_output, target_output,weights=mask)

    loss = tf.reduce_mean(att_loss)

    



    return loss,train_decode_result, pred_decode_result

