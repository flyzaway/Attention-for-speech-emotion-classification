# Attention-for-speech-emotion-classification

----
使用LSTM-Attention/GRU-Attention在Casia库中实现语音情感识别
----
## 文件介绍
1. casia 按情感类型放入数据，casia全库需要保密，这里不予提供
2. logs 记录训练过程，方便tensorboard查看
3. model 存放当前最好模型
4. pkl 存放提取好的特征
5. attention_LSTM.py 实现了attention层
6. Predict_epoch.py 实现了一个keras的callback函数，在每一层训练时都计算一次分类结果，方法统计最好结果时的混淆矩阵
7. analysis.py 对attention层学习到的特征与lstm层学习到的特征进行可视化
8. test_gru,test_lstm,Bilstm,BiGRU使用4个模型实现语音情感识别
9. feature 使用两个语音方面的库进行特征提取，python_speech_features、librosa
10. get_feature 批量获取所以语音样本的特征
11. cross_validate 实现了交叉验证和非交叉验证的方式来准备训练测试数据
