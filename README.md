# 用CRNN+seq2seq+attention识别中文
中文类别为5990类，类别可以在char_std_5990找到，中文样本的合成参考的是caffe_ocr项目，
## 网络结构
CNN用的是CRNN中的结构，一层双向lstm做编码器，一层GRU做解码器
## 如何训练
训练需要2个txt文件（train.txt，test.txt）保存图片的名字以及label,<br>
可以在这里下载样本[(caffe_ocr)百度网盘](https://pan.baidu.com/s/1dFda6R3#list/path=%2F)<br>
在config.py里修改路径，运行train.py
## 依赖
TensorFlow >=1.2<br>
opencv<br>
## 引用
[caffe_ocr](https://github.com/senlinuc/caffe_ocr)<br>
[attention-ocr-toy-example](https://github.com/ray075hl/attention-ocr-toy-example)
