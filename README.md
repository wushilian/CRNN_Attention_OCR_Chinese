# CRNN_Attention_OCR
CRNN with attention to do OCR
## network model
CRNN is base CNN,and BiLSTM with 256 hidden_units is encode network ,GRU with 256 hidden_units is decode network

## how to use
put your image in 'train' dir,and image name should be like "xx_label_xx.jpg",Parameters are set in config.py,and then just run the train.py

## Dependency Library
TensorFlow >=1.2
opencv
