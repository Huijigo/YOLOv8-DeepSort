# 项目简介：
使用YOLOv8+Deepsort实现行人，代码封装成一个Detector类，更容易嵌入到自己的项目中。

# 权重简介
在./deep_sort/configs/deep_sort.yaml文件目录的 REID_CKPT属性中指定DeepSort特征提取网络的权重
在AIDetector_pytorch_Change这个目录中存放了yolov8的网络权重加载

# 训练方式
单独训练好DeepSort的特征提取网络，再单独训练YoloV8的权重网络

# 测试方式
更换上述权重简介中权重位置即可在detect.py中实现detect

# 评价指标
如何评价模型的训练结果，请参考下面的网页信息\\

https://blog.csdn.net/weixin_44238733/article/details/124148469?spm=1001.2014.3001.5506