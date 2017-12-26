import numpy as np
import cv2
import os
learning_rate=0.001
momentum=0.9
START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
VOCAB = {'<GO>': 0, '<EOS>': 1, '<UNK>': 2,'<PAD>':3}#分别表示开始，结束，未出现的字符
VOC_IND={}
charset='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(len(charset)):
    VOCAB[charset[i]]=i+4
for key in VOCAB:
    VOC_IND[VOCAB[key]]=key

MAX_LEN_WORD=27#标签的最大长度，以PAD
VOCAB_SIZE = len(VOCAB)
BATCH_SIZE = 40
RNN_UNITS = 256
EPOCH=10000
IMAGE_WIDTH=120
IMAGE_HEIGHT = 32
MAXIMUM__DECODE_ITERATIONS = 20
DISPLAY_STEPS = 2
LOGS_PATH = 'log'
CKPT_DIR = 'save_model'
train_dir='train'
is_restore=True
def label2int(label):#label shape (num,len)
    #seq_len=[]
    target_input=np.ones((len(label), MAX_LEN_WORD), dtype=np.float32) +2#初始化为全为PAD
    target_out = np.ones((len(label), MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
    for i in range(len(label)):
        #seq_len.append(len(label[i]))
        target_input[i][0]=0#第一个为GO
        for j in range(len(label[i])):
            target_input[i][j+1]=VOCAB[label[i][j]]
            target_out[i][j]=VOCAB[label[i][j]]
        target_out[i][len(label[i])]=1
    return target_input,target_out
def int2label(decode_label):
    label=[]
    for i in range(decode_label.shape[0]):
        temp=''
        for j in range(decode_label.shape[1]):
            if VOC_IND[decode_label[i][j]]=='<EOS>':
                break
            elif decode_label[i][j]==3:
                continue
            else:
                temp+=VOC_IND[decode_label[i][j]]
        label.append(temp)
    return  label

def read_data(data_dir):
    image = []
    labels = []
    num=0
    for root, sub_folder, file_list in os.walk(data_dir):
        for file_path in file_list:
            image_name = os.path.join(root, file_path)
            im = cv2.imread(image_name, 0)  # /255.#read the gray image
            img = cv2.resize(im, (IMAGE_WIDTH,IMAGE_HEIGHT))
            img = img.swapaxes(0, 1)
            image.append(np.array(img[:, :, np.newaxis]))
            labels.append(image_name.split('_')[1])
            num+=1
    print('---------------------------------get image:',num)
    return np.array(image),labels
def cal_acc(pred,label):
    num=0
    for i in range(len(pred)):
        if pred[i]==label[i]:
            num+=1
    return num*1.0/len(pred)

