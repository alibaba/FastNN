# FastNN模型库([English version](https://github.com/alibaba/FastNN/blob/master/README_en.md))
## 1. 简介
FastNN（Fast Neural Networks）是一个基于[PAISoar](https://yq.aliyun.com/articles/705132)实现分布式训练的基础算法库，当前FastNN只支持计算机视觉的部分经典算法，后续会逐步开放更多的先进模型。如需在机器学习平台PAI（Platform of Artificial Intelligence）试用FastNN分布式训练服务，可访问[PAI平台官方主页](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf)开通，即可在PAI Studio或DSW-notebook上提交分布式机器学习任务，具体操作流程可参考[TensorFlow组件使用手册](https://help.aliyun.com/document_detail/49571.html?spm=a2c4g.11186623.6.579.10501312JxztvO)、[FastNN-on-PAI-Studio使用手册](https://help.aliyun.com/document_detail/139435.html)及“[快速试用](https://github.com/alibaba/FastNN#2-%E5%BF%AB%E9%80%9F%E8%AF%95%E7%94%A8)”。

FastNN功能简介如下：
* 模型支持 

    a.部分经典计算机视觉模型，包括inception、resnet、mobilenet、vgg、alexnet、nasnet等；
    
    b.后续会增加计算机视觉、自然语言处理等领域的其他先进模型；

* 数据并行训练（需要开通机器学习平台PAI服务，在PAI Studio或DSW-notebook试用）

    a.单机多卡
    
    b.多机多卡

* 半精度训练

* 模型训练类型

    a.模型预训练
    
    b.模型调优：默认只重载可训练的变量，如需对checkpoint选择性重载，可修改images/utils/misc_utils.py的get_assigment_map_from_checkpoint函数

我们选择ResNet-v1-50模型上在阿里云集群机器(GPU卡型为P100)上进行了大规模测试。从测试数据来看，PAISoar加速效果都非常理想，都能够取得接近线性scale的加速效果。

![resnet_v1_50](http://pai-online.oss-cn-shanghai.aliyuncs.com/fastnn-data/readme/resnet_v1_50.png)


## 2. 快速试用
本章节旨在给出FastNN库中已有模型的试用说明，具体试用流程分为两步：
* 数据准备：包括本地数据准备和PAI Web数据准备；
* 训练启动：包括本地执行脚本的编辑和PAI Web任务参数设置。

### 2.1  数据准备
为了方便试用FastNN算法库images目录下的计算机视觉模型，我们准备好了一些公开数据集或相应下载及格式转换脚本，包括图像数据cifar10、mnist以及flowers。
#### 2.1.1 本地数据
借鉴TF-Slim库中提供数据下载及格式转换脚本（images/datasets/download_and_convert_data.py），以cifar10数据为例，脚本如下：
```
DATA_DIR=/tmp/data/cifar10
python download_and_convert_data.py \
	--dataset_name=cifar10 \
	--dataset_dir="${DATA_DIR}"
```
脚本执行完毕后，在/tmp/data/cifar10目录下有以下tfrecord文件：
```
$ ls ${DATA_DIR}
cifar10_train.tfrecord
cifar10_test.tfrecord
labels.txt
```

#### 2.1.2 OSS数据
为了方便在[PAI平台](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf)试用FastNN，cifar10、mnist、flowers数据已下载并转换为tfrecord后存储在公开oss上，可通过阿里机器学习平台PAI的“读取文件数据”组件访问。存储oss路径如下：

|数据集|分类数|训练集|测试集|存储路径|
| :-----: | :----: | :-----:| :----:| :---- |
| mnist   |  10    |  3320  | 350   | 北京：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/mnist/ 上海：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/mnist/
| cifar10 |  10    | 50000  |10000  | 北京：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/cifar10/ 上海：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/cifar10/
| flowers |  5     |60000   |10000  | 北京：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/flowers/ 上海：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/flowers/

### 2.2 训练启动
FastNN模型库主文件为train_image_classifiers.py，其中训练脚本最常用的有以下六个参数：
* task_type：字符串类型。取值为“pretrain”、“finetune”之一，指出任务类型为模型预训练或模型调优；
* enable_paisoar：布尔类型。默认True，本地试用时需置为False。
* dataset_name：字符串类型。默认mock，指出训练数据解析文件，如images/datasets目录下的cifar10.py、flowers.py、mnist.py文件。
* train_files：字符串类型。默认None，以“,”为分隔符表示所有训练文件。
* dataset_dir：字符串类型。默认None，指出训练数据路径。
* model_name：字符串类型。默认inception_resnet_v2，指明模型名称，包括resnet_v1_50、vgg、inception等，详见images/models目录下的所有模型。
特别地，当task_type=finetune时，需额外指定model_dir、ckpt_file_name参数，分别指明模型checkpoint路径及checkpoint文件名。
下面分为“本地试用”、“PAI平台运行”两个章节详述试用方法。

#### 2.2.1 本地试用
本地不支持PAISoar功能，即不支持分布式训练。若只需要本地试用FastNN模型库的单机单卡场景下的训练，需要在执行脚本中设置用户参数enable_paisoar=False，另外有以下软件需求：

|软件|版本|
| :-----: | :----: |
|python|>=2.7.6|
|TensorFlow|>=1.8|
|CUDA|>= 9.0|
|cuDNN| >= 7.0|

下面以Resnet-v1-50模型在cifar10数据训练为例梳理试用流程。
##### 2.2.1.1 预训练脚本

```
DATASET_DIR=/tmp/data/cifar10
TRAIN_FILES=cifar10_train.tfrecord
python train_image_classifiers.py \
	--task_type=pretrain \ 
	--enable_paisoar=False \
	--dataset_name=cifar10 \
	--train_files="${TRAIN_FILES}" \
	--dataset_dir="${DATASET_DIR}" \
	--model_name=resnet_v1_50
```
##### 2.2.1.2 模型调优脚本

```
MODEL_DIR=/path/to/model_ckpt
CKPT_FILE_NAME=resnet_v1_50.ckpt
DATASET_DIR=/tmp/data/cifar10
TRAIN_FILES=cifar10_train.tfrecord
python train_image_classifiers.py \
	--task_type=finetune \
	--enable_paisoar=False \
	--dataset_name=cifar10 \
	--train_files="${TRAIN_FILES}" \
	--dataset_dir="${DATASET_DIR}" \
	--model_name=resnet_v1_50 \
	--model_dir="${MODEL_DIR}" \
	--ckpt_file_name="${CKPT_FILE_NAME}"
```

#### 2.2.2 PAI平台运行
机器学习平台PAI目前支持的框架包括 TensorFlow（兼容开源TF1.4、1.8版本），MXNet 0.9.5， Caffe rc3。TensorFlow 和 MXNet 支持用户自己编写的 Python 代码， Caffe 支持用户自定义网络文件。其中tensorflow框架内置PAISoar功能，支持单机多卡、多机多卡的分布式训练。如需在PAI平台试用FastNN，请参考[FastNN-On-PAI](https://help.aliyun.com/document_detail/139435.html)文档。

## 3. 用户参数指南
FastNN库综合各个模型及PAISoar框架的需求，统一将可能用到的超参定义在flags.py文件（支持用户自定义新超参）中，参数可分为以下六大类：

* 数据集参数：确定训练集的基本属性，如训练集存储路径dataset_dir；
* 数据预处理参数：数据预处理函数及dataset pipeline相关参数；
* 模型参数：模型训练基本超參，包括model_name、batch_size等；
* 学习率参数：学习率及其相关调优参数；
* 优化器参数：优化器及其相关参数；
* 日志参数：关于输出日志的参数；
* 性能调优参数：混合精度等其他调优参数。

### 3.1 数据集参数

|#名称|#类型|#描述|
| :-----: | :----: | :-----|
|dataset_name|string|指定输入数据名称，默认mock|
|dataset_dir|string|指定本地输入数据集路径，默认为None|
|num_sample_per_epoch|integer|数据集总样本数|
|num_classes|integer|数据分类数，默认100|
|train_files|string|训练数据文件名，文件间分隔符为逗号，如"0.tfrecord,1.tfrecord"|

### 3.2 数据预处理参数

|#名称|#类型|#描述|
| :-----: | :----: | :-----|
|preprocessing_name|string|预处理方法名，默认None|
|shuffle_buffer_size|integer|样本粒度进行shuffle的buffer大小，默认1024|
|num_parallel_batches|integer|与batch_size乘积为map_and_batch的并行线程数，默认8|
|prefetch_buffer_size|integer|预取N个batch数据，默认N=32|
|num_preprocessing_threads|integer|预取线程数，默认为16|
|datasets_use_caching|bool|以内存为开销进行输入数据的压缩缓存|

### 3.3 模型参数

|#名称|#类型|#描述|
| :-----: | :----: | :-----|
|task_type|string|支持模型预训练（pretrain）和模型调优（finetune）, 默认取值pretrain|
|model_name|string|指定模型名称，默认inception_resnet_v2|
|num_epochs|integer|训练epochs，默认100|
|weight_decay|float|模型权重衰减系数, 默认0.00004|
|max_gradient_norm|float|根据全局归一化值进行梯度裁剪, 默认取值为None，不进行梯度裁剪|
|batch_size|integer|批大小, 默认32|
|model_dir|string|checkpoin所在路径，默认为None|
|ckpt_file_name|string|checkpoint文件名，默认为None|

### 3.4 学习率参数

|#名称|#类型|#描述|
| :-----: | :----: | :-----|
|warmup_steps|integer|逆衰减学习率的迭代数. 默认0.|
|warmup_scheme|string|学习率逆衰减的方式. 可选:'t2t' 指Tensor2Tensor, 初始化为指定学习率的1/100，然后exponentiate逆衰减到指定学习率为止|
|decay_scheme|string|学习率衰减的方式. 可选:1、luong234: 在2/3的总迭代数之后, 开始4次衰减，衰减系数为1/2; 2、luong5: 在1/2的总迭代数之后, 开始5次衰减，衰减系数为1/2; 3、luong10: 在1/2的总迭代数之后, 开始10次衰减，衰减系数为1/2.|
|learning_rate_decay_factor|float|学习率衰减系数, 默认0.94|
|learning_rate_decay_type|string|学习率衰减类型. 可选["fixed", "exponential", "polynomial"], 默认exponential|
|learning_rate|float|学习率初始值，默认0.01|
|end_learning_rate|float|衰减时学习率值的下限，默认0.0001|

### 3.5 优化器参数

|#名称|#类型|#描述|
| :-----: | :----: | :-----|
|optimizer|string|优化器名称, 取值["adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd"，"rmsprop"]. 默认"rmsprop"|
|adadelta_rho|float|adadelta的衰减系数, default 0.95, specially for Adadelta|
|adagrad_initial_accumulator_value|float|AdaGrad积累器的起始值, 默认0.1, Adagrada优化器专用参数|
|adam_beta1|float|一次动量预测的指数衰减率, 默认0.9, Adam优化器专用参数|
|adam_beta2|float|二次动量预测的指数衰减率, 默认0.999, Adam优化器专用参数|
|opt_epsilon|float|优化器偏置值, 默认1.0, Adam优化器专用参数|
|ftrl_learning_rate_power|float|学习率参数的幂参数, 默认-0.5, Ftrl优化器专用参数|
|ftrl_initial_accumulator_value|float|FTRL积累器的起始, 默认0.1, Ftrl优化器专用参数|
|ftrl_l1|float|FTRL l1正则项, 默认0.0, Ftrl优化器专用参数|
|ftrl_l2|float|FTRL l2正则项, 默认0.0, Ftrl优化器专用参数|
|momentum|float|MomentumOptimizer的动量参数, 默认0.9, Momentum优化器专用参数|
|rmsprop_momentum|float|RMSPropOptimizer的动量参数, 默认0.9|
|rmsprop_decay|float|RMSProp的衰减系数, 默认0.9|

### 3.6 日志参数

|#名称|#类型|#描述|
| :-----: | :----: | :-----|
|stop_at_step|integer|训练总迭代数, 默认100|
|log_loss_every_n_iters|integer|打印loss信息的迭代频率, 默认10|
|profile_every_n_iters|integer|打印timeline的迭代频率, 默认0|
|profile_at_task|integer|输出timeline的机器对应索引, 默认0，对应chief worker|
|log_device_placement|bool|是否输出device placement信息, 默认False|
|print_model_statistics|bool|是否输出可训练变量信息, 默认false|
|hooks|string|指定训练hooks, 默认"StopAtStepHook,ProfilerHook,LoggingTensorHook,CheckpointSaverHook"|

### 3.7 性能调优参数

|#名称|#类型|#描述|
| :-----: | :----: | :-----|
|use_fp16|bool|是否进行半精度训练, 默认True|
|loss_scale|float|训练中loss值scale的系数, 默认1.0|
|enable_paisoar|bool|是否使用paisoar，默认True.|
|protocol|string|默认grpc.rdma集群可用“grpc+verbs”提升数据存取效率|

## 4. 如何实现自定义需求
若已有模型等实现满足不了需求，可通过继承dataset／models／preprocessing接口进一步开发。在此之前需要了解fastnn库的基本流程(以images为例，代码入口文件为train_image_classifiers.py)，整体代码架构流程如下:

```
# 初始化models中某模型得到network_fn，并可能返回输入参数train_image_size;
    network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=FLAGS.num_classes,
            weight_decay=FLAGS.weight_decay,
            is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
# 若用户指定参数FLAGS.preprocessing_name，则初始化数据预处理函数得到preprocess_fn;
    preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.model_name or FLAGS.preprocessing_name,
                is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
# 用户指定dataset_name，选择正确的tfrecord格式，同步调用preprocess_fn解析数据集得到数据dataset_iterator;
    dataset_iterator = dataset_factory.get_dataset_iterator(FLAGS.dataset_name,
                                                            train_image_size,
                                                            preprocessing_fn,
                                                            data_sources,
# 根据network_fn、dataset_iterator，定义计算loss的函数loss_fn：
    def loss_fn():
    	with tf.device('/cpu:0'):
      		images, labels = dataset_iterator.get_next()
        logits, end_points = network_fn(images)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(logits, tf.float32), weights=1.0)
        if 'AuxLogits' in end_points:
          loss += tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(end_points['AuxLogits'], tf.float32), weights=0.4)
        return loss
# 调用PAI-Soar API封装loss_fn、tf原生optimizer
    opt = paisoar.ReplicatedVarsOptimizer(optimizer, clip_norm=FLAGS.max_gradient_norm)
    loss = optimizer.compute_loss(loss_fn, loss_scale=FLAGS.loss_scale)
# 依据opt和loss形式化定义training tensor
    train_op = opt.minimize(loss, global_step=global_step)
```

因此，为了进一步开发新需求，开发者需要了解dataset、models、preprocessing三个接口的使用。

### 4.1 增加数据集API
**FastNN库已实现支持读取tfrecord格式的数据**，并基于TFRecordDataset接口实现dataset pipeline以供模型训练试用，几乎可掩盖数据预处理时间。另外，目前实现逻辑在数据分片实现不够精细，要求用户在数据准备时尽量保证数据能平均分配到每台机器，即：
* 每个tfreocrd文件的样本数量几乎一样；
* 每个worker处理的tfrecord文件数几乎一致。

若数据格式同为tfrecord，可参考datasets目录下cifar10/mnist/flowers各文件等，以cifar10数据为例：
* 假设cifar10数据的key_to_features格式为：
```
features={
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
}
```
* 在datasets目录下创建数据解析文件cifar10.py，并编辑内容示例如下:
```
"""Provides data for the Cifar10 dataset.
The dataset scripts used to create the dataset can be found at:
utils/scripts/data/download_and_convert_cifar10.py
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
"""Expect func_name is ‘parse_fn’
"""
def parse_fn(example):
  with tf.device("/cpu:0"):
    features = tf.parse_single_example(
      example,
      features={
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      }
    )
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = features['image/class/label']
    return image, label
```

* 在datasets/dataset_factory.py补足dataset_map
```
from datasets import cifar10
datasets_map = {
    'cifar10': cifar10,
}
```
* 执行任务脚本时，可通过指定参数dataset_name=cifar10和train_files=cifar10_train.tfrecord使用cifar10数据进行模型训练。

如需读取其他数据格式，需要自行实现dataset pipeline构建逻辑（参考utils/dataset_utils.py）。

### 4.2 增加模型API
开发者如需完成新模型的开发，参考images/models目录下各模型代码，有以下要求：

* 单机单卡代码跑通，并能正常收敛；
* 单机单卡代码暴露出输入／输出接口，输入属性包括Type／Shape等对应于preprocessing输出属性，以及输出属性包括Type／Shape等对应数据集label的属性；
* 没有实现任何分布式逻辑。

datasets、preprocessing逻辑在仅开发新模型时可直接复用，关于模型需要做以下操作：

* models/model_factory.py中models_map和arg_scopes_map增加新模型的定义，可参考model_factory.py中对已有模型的定义；
* 正常收敛的新模型代码导入到project的images/models目录即可。

### 4.3 增加数据预处理API
开发者如需自定义新的数据预处理流程，有以下要求：

* 暴露出输入／输出接口，输入的属性（包括Type／Shape等）对应于dataset输出的属性，以及输出属性（包括Type／Shape等）对应算法模型输入的属性；

具体步骤如下：

* preprocessing_factory.py中preprocessing_fn_map增加新的预处理函数的定义；

* 预处理函数代码导入到project的images/preprocessing下即可，可参考已有实现，如inception_preprocessing.py等

### 4.4 增加损失函数API
FastNN库images的主文件train_image_classifiers中默认使用tf.losses.sparse_softmax_cross_entropy构造loss_fn，若需要构造自定义的损失函数，直接修改主文件中的loss_fn。值得注意的是，使用PAISoar做分布式时，loss_fn要求只返回loss，若需返回自定义参数，可通过全局参数传输，可参考主文件中loss_fn返回accuracy。

## 5. 总结与致谢
在本手册里面我们简单的介绍了阿里机器学习平台PAI在FastNN算法库的一些工作。我们深信这些工作能够帮助算法同学快速的进行模型的迭代、支持更大规模的样本训练和更大空间上的模型创新的想象力。接下来PAI还会持续的在分布式上进行更多的创新工作，包括更优性能的模型并行、通信梯度压缩等。

FastNN是基于阿里机器学习平台PAI的PAISoar组件实现分布式训练的基础算法库，包含计算机视觉领域目前state-of-art的经典模型，后续会支持NLP等领域的模型，包括Bert、XLNet、GPT-2、NMT等。目前FastNN聚焦于模型训练分布式性能的加速，其中images模型直接引用[TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim#tensorflow-slim-image-classification-model-library)的实现。感谢TensorFlow社区贡献计算机视觉领域经典模型的开源实现，如有不妥之处，敬请指正。
