# FastNN Model Library
## 1. 简介
FastNN（Fast Neural Networks）是一个基于[PAISoar](https://yq.aliyun.com/articles/705132)实现的分布式训练的基础算法库。目前功能简介如下：
* 模型类别

    a.部分经典计算机视觉模型，包括inception、resnet、mobilenet、vgg、alexnet、nasnet等；
    
    b.其他类型模型后续会陆续添加；
    
* 数据并行训练（需要开通机器学习平台PAI服务，在PAI Studio或DSW-notebook试用）

    a.单机多卡
    
    b.多机多卡
    
* 混合精度训练
* 模型训练类型

    a.模型预训练
    
    b.模型调优：默认只restore trainable variables，如需自定义对checkpoint选择性restore，可修改image_models/utils/misc_utils.py的get_assigment_map_from_checkpoint函数

目前FastNN只包括计算机视觉的部分经典模型，后续会逐步开放NLP等领域的State-of-Art模型。如需试用机器学习平台PAI（Platform of Artificial Intelligence）服务，可访问[PAI平台官方主页](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf)开通，即可在PAI Studio或DSW-notebook上提交机器学习任务，具体操作流程可参考[TensorFlow使用手册](https://help.aliyun.com/document_detail/49571.html?spm=a2c4g.11186623.6.579.10501312JxztvO)。

我们针对ResNet-v1-50模型上在弹内弹外集群P100上进行了大规模测试。从测试数据来看，PAISoar加速效果都非常理想，都能够取得接近线性scale的加速效果。
![resnet_v1_50](https://intranetproxy.alipay.com/skylark/lark/0/2019/png/62136/1566391007222-82bea78a-4462-4540-b6af-bca9ae9de74c.png)

## 2.  数据准备
为了方便试用FastNN算法库image_models目录下的CV模型，我们准备好了一些公开数据集及其相应download_and_convert脚本，包括图像数据cifar10、mnist以及flowers。
### 2.1 本地数据
借鉴TF-Slim库中提供数据下载及格式转换脚本（image_models/datasets/download_and_convert_data.py），以cifar10数据为例，脚本如下：
```python
DATA_DIR=/tmp/data/cifar10
python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir="${DATA_DIR}"
```
脚本执行完毕后，在/tmp/data/cifar10目录下有以下tfrecord文件：
>$ ls ${DATA_DIR}

>cifar10_train.tfrecord

>cifar10_test.tfrecord

>labels.txt

### 2.2 OSS数据
为了方便在[PAI平台](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf)试用FastNN，cifar10、mnist、flowers数据已下载并转换为tfrecord后存储在公开oss上，可通过机器学习平台PAI的“读取文件数据”访问，存储oss路径如下：

|数据集|分类数|训练集|测试集|存储路径|
| :-----: | :----: | :-----:| :----:| :---- |
| mnist   |  10    |  3320  | 350   | 北京：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/mnist/ 上海：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/mnist/
| cifar10 |  10    | 50000  |10000  | 北京：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/cifar10/ 上海：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/cifar10/
| flowers |  5     |60000   |10000  | 北京：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/flowers/ 上海：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/flowers/

## 3. 如何运行
### 3.1 本地试用
本地不支持PAISoar功能，即不支持分布式训练。若只需要本地试用FastNN模型库的单机单卡场景下的训练，需要在执行脚本中设置用户参数enable_paisoar=False，下面以Resnet-v1-50模型在cifar10数据训练为例梳理测试流程。
#### 3.1.1 Pretrain脚本

```python
DATASET_DIR=/tmp/data/cifar10
TRAIN_FILES=cifar10_train.tfrecord
python train_image_classifiers.py \
	--task_type=pretrain \ 
	--enable_paisoar=False \
    --dataset_name=cifar10 \
    --train_files=${TRAIN_FILES} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50
```
#### 3.1.2 Finetune脚本

```python
MODEL_DIR=/path/to/model_ckpt
CKPT_FILE_NAME=resnet_v1_50.ckpt
DATASET_DIR=/tmp/data/cifar10
TRAIN_FILES=cifar10_train.tfrecord
python train_image_classifiers.py \
	--task_type=finetune \
	--enable_paisoar=False \
    --dataset_name=cifar10 \
    --train_files=${TRAIN_FILES} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50 \
    --model_dir=${MODEL_DIR} \
    --ckpt_file_name=${CKPT_FILE_NAME}
```

### 3.2 PAI平台运行
机器学习平台PAI目前支持的框架包括 TensorFlow（兼容开源TF1.4、1.8版本），MXNet 0.9.5， Caffe rc3。TensorFlow 和 MXNet 支持用户自己编写的 Python 代码， Caffe 支持用户自定义网络文件。其中tensorflow框架内置PAISoar功能，支持单机多卡、多机多卡的分布式训练，具体使用参考[FastNN-On-PAI](https://yuque.antfin-inc.com/docs/share/1368e10c-45f1-443e-88aa-0bb5425fea72)文档。

## 4. 用户参数指南
3.2.4节中给出的用户参数文件示例仅给出了部分参数，FastNN库综合各个模型及PAISoar框架的需求，统一将可能用到的超參定义保存在flags.py文件（支持用户自定义新超參）中，已定义参数具体可分为以下部分。

* Dataset Option：确定训练集的基本属性，如训练集存储路径dataset_dir；
* Dataset PreProcessing Option：数据预处理函数及dataset pipeline相关参数；
* Model Params Option：模型训练基本超參，包括model_name、batch_size等；
* Learning Rate Tuning：学习率及其相关调优参数；
* Optimizer Option：优化器及其相关参数；
* Logging Option：关于输出Log的参数；
* Performance Tuning：混合精度等其他调优参数。

### 4.1 Dataset Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|dataset_name|string|指定输入数据名称，默认mock|
|dataset_dir|string|指定本地输入数据集路径，默认为None|
|num_sample_per_epoch|integer|数据集总样本数|
|num_classes|integer|数据lable数，默认100|
|train_files|string|训练数据文件名，文件间分隔符为逗号，如"0.tfrecord,1.tfrecord"|

### 4.2 Dataset Preprocessing Tuning

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|preprocessing_name|string|预处理方法名，默认None|
|shuffle_buffer_size|integer|数据shuffle的buffer size，默认1024|
|num_parallel_batches|integer|与batch_size乘积为map_and_batch的并行线程数，默认8|
|prefetch_buffer_size|integer|预取N个batch数据，默认N=32|
|num_preprocessing_threads|integer|预取线程数，默认为16|
|datasets_use_caching|bool|Cache the compressed input data in memory. This improves the data input performance, at the cost of additional memory|

### 4.3 Model Params Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|task_type|string|support pretrain or finetune, default pretrain|
|model_name|string|指定模型名称，默认inception_resnet_v2|
|num_epochs|integer|训练epochs，默认100|
|weight_decay|float|The weight decay on the model weights, default 0.00004|
|max_gradient_norm|float|clip gradient to this global norm, default None for clip-by-global-norm diabled|
|batch_size|integer|The number of samples in each batch, default 32|
|model_dir|string|dir of checkpoint for init|
|ckpt_file_name|string|Initial checkpoint (pre-trained model: base_dir + model.ckpt).|

### 4.4 Learning Rate Tuning

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|warmup_steps|integer|how many steps we inverse-decay learning. default 0.|
|warmup_scheme|string|how to warmup learning rates. Options include:'t2t' refers to Tensor2Tensor way, start with lr 100 times smaller,then exponentiate until the specified lr. default 't2t'|
|decay_scheme|string|How we decay learning rate. Options include:1、luong234: after 2/3 num train steps, we start halving the learning rate for 4 times before finishing;2、luong5: after 1/2 num train steps, we start halving the learning rate for 5 times before finishing;3、luong10: after 1/2 num train steps, we start halving the learning rate for 10 times before finishing.|
|learning_rate_decay_factor|float|learning rate decay factor, default 0.94|
|learning_rate_decay_type|string|specifies how the learning rate is decayed. One of ["fixed", "exponential", or "polynomial"], default exponential|
|learning_rate|float|学习率初始值，默认0.01|
|end_learning_rate|float|decay时学习率值的下限，默认0.0001|

### 4.5 Optimizer Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|optimizer|string|the name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop". Default "rmsprop"|
|adadelta_rho|float|the decay rate for adadelta, default 0.95, specially for Adadelta|
|adagrad_initial_accumulator_value|float|starting value for the AdaGrad accumulators, default 0.1, specially for Adagrada|
|adam_beta1|float|the exponential decay rate for the 1st moment estimates, default 0.9, specially for Adam|
|adam_beta2|float|the exponential decay rate for the 2nd moment estimates, default 0.999, specially for Adam|
|opt_epsilon|float|epsilon term for the optimizer, default 1.0, specially for Adam|
|ftrl_learning_rate_power|float|the learning rate power, default -0.5, specially for Ftrl|
|ftrl_initial_accumulator_value|float|Starting value for the FTRL accumulators, default 0.1, specially for Ftrl|
|ftrl_l1|float|The FTRL l1 regularization strength, default 0.0, specially for Ftrl|
|ftrl_l2|float|The FTRL l2 regularization strength, default 0.0, specially for Ftrl|
|momentum|float|The momentum for the MomentumOptimizer, default 0.9, specially for Momentum|
|rmsprop_momentum|float|Momentum for the RMSPropOptimizer, default 0.9|
|rmsprop_decay|float|Decay term for RMSProp, default 0.9|

### 4.6 Logging Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|stop_at_step|integer|the whole training steps, default 100|
|log_loss_every_n_iters|integer|frequency to print loss info, default 10|
|profile_every_n_iters|integer|frequency to print timeline, default 0|
|profile_at_task|integer|node index to output timeline, default 0|
|log_device_placement|bool|whether or not to log device placement, default False|
|print_model_statistics|bool|whether or not to print trainable variables info, default false|
|hooks|string|specify hooks for training, default "StopAtStepHook,ProfilerHook,LoggingTensorHook,CheckpointSaverHook"|

### 4.7 Performanse Tuning Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|use_fp16|bool|whether to train with fp16, default True|
|loss_scale|float|loss scale value for training, default 1.0|
|enable_paisoar|bool|whether or not to use pai soar，default True.|
|protocol|string|default grpc.For rdma cluster, use grpc+verbs instead|

## 5. 如何实现自定义需求
若已有模型满足不了用户需求，可通过继承dataset／models／preprocessing接口，在进一步开发之前需要了解fastnn库的基本流程(以image_models为例，代码入口文件为train_image_classifiers.py):

```python
整体代码架构流程如下：
- 初始化models中某模型得到network_fn，并可能返回输入参数train_image_size;
    network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=FLAGS.num_classes,
            weight_decay=FLAGS.weight_decay,
            is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
- 若用户指定参数FLAGS.preprocessing_name，则初始化数据预处理函数得到preprocess_fn;
    preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.model_name or FLAGS.preprocessing_name,
                is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
- 用户指定dataset_name，选择正确的tfrecord格式，同步调用preprocess_fn解析数据集得到数据dataset_iterator;
	dataset_iterator = dataset_factory.get_dataset_iterator(FLAGS.dataset_name,
                                                            train_image_size,
                                                            preprocessing_fn,
                                                            data_sources,
- 根据network_fn、dataset_iterator，定义计算loss的函数loss_fn：
  	def loss_fn():
    	with tf.device('/cpu:0'):
      		images, labels = dataset_iterator.get_next()
        logits, end_points = network_fn(images)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(logits, tf.float32), weights=1.0)
        if 'AuxLogits' in end_points:
          loss += tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(end_points['AuxLogits'], tf.float32), weights=0.4)
        return loss
- 调用PAI-Soar API封装loss_fn、tf原生optimizer
	opt = paisoar.ReplicatedVarsOptimizer(optimizer, clip_norm=FLAGS.max_gradient_norm)
	loss = optimizer.compute_loss(loss_fn, loss_scale=FLAGS.loss_scale)
- 依据opt和loss形式化定义training tensor
    train_op = opt.minimize(loss, global_step=global_step)
```

因此，为了进一步开发新需求，开发者需要了解dataset、models、preprocessing三个接口的使用。

### 5.1 dataset
**FastNN库已实现支持读取tfrecord格式的数据**，并基于TFRecordDataset接口实现dataset pipeline以供模型训练试用，如需读取其他数据格式，需要自行实现该部分逻辑（参考utils/dataset_utils.py）。另外，目前实现逻辑在数据分片实现不够精细，仅保证每个worker处理的tfrecord文件数尽量一致，并要求训练集的文件数不少于worker数，以保证每个worker处理不同的数据，其中cifar10、mnist训练集因只有一个文件仅支持单机多卡分布式训练，故要求用户在数据准备时尽量保证数据能平均分配到每台机器。

若数据格式同为tfrecord，只需操作如下（可参考datasets目录下cifar10/mnist/flowers各文件等）：
* 要求自定义的tfrecord格式文件名和用户参数指定dataset_name一致， 如dataset_name=cifar10，则在datasets目录下新建cifar10.py，并编辑内容示例如下:

```python
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
* 执行任务脚本时，用户需要传參train_files（含义相见4.1章节介绍）

### 5.2 models
开发者如需完成新模型的开发，参考image_models/models目录下各模型代码，有以下要求：

* 单机单卡代码跑通，并能正常收敛；
* 单机单卡代码暴露出输入／输出接口，输入属性包括Type／Shape等对应于preprocessing输出属性，以及输出属性包括Type／Shape等对应数据集label的属性；
* 没有实现任何分布式逻辑。

datasets、preprocessing逻辑在仅开发新模型时可直接复用，关于模型需要做以下操作：

* models/model_factory.py中models_map和arg_scopes_map增加新模型的定义，可参考model_factory.py中对已有模型的定义；
* 正常收敛的新模型代码导入到project的image_models/models目录即可。

### 5.3 preprocessing
开发者如需自定义新的数据预处理流程，有以下要求：

* 暴露出输入／输出接口，输入的属性（包括Type／Shape等）对应于dataset输出的属性，以及输出属性（包括Type／Shape等）对应算法模型输入的属性；

具体步骤如下：

* preprocessing_factory.py中preprocessing_fn_map增加新的预处理函数的定义；

* 预处理函数代码导入到project的image_models/preprocessing下即可，可参考已有实现，如inception_preprocessing.py等

## 6. 总结与致谢
在本手册里面我们简单的介绍了阿里机器学习平台PAI在FastNN算法库的一些工作。我们深信这些工作能够帮助算法同学快速的进行模型的迭代、支持更大规模的样本训练和更大空间上的模型创新的想象力。接下来PAI还会持续的在分布式上进行更多的创新工作，包括更优性能的模型并行、通信梯度压缩等。

FastNN是基于阿里机器学习平台PAI的PAISoar组件实现分布式训练的基础算法库，包含计算机视觉领域目前state-of-art的经典模型，后续会支持NLP等领域的模型，包括Bert、XLNet、GPT-2、NMT等。目前FastNN聚焦于模型训练分布式性能的加速，其中image_models模型直接引用[TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim#tensorflow-slim-image-classification-model-library)的实现。感谢TensorFlow社区贡献计算机视觉领域经典模型的开源实现，如有不妥之处，敬请指正。