# FastNN Model Library
## 1. Introduction
FastNN (Fast Neural Networks) is a package of models that aims to show how to implement distributed model in TensorFlow based on [PAISoar](https://yq.aliyun.com/articles/705132). FastNN could help researchers effectively apply distributed neural networks. For now, the initial version of FastNN  provides several classic models on Computer Vision (CV). More state-of-art models would be also integrated into the package in the near future.

If you have interest to try running FastNN models in distributed environment on Alibaba PAI (Platform of Artificial Intelligence), please turn to [PAI Homepage](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf), then submit machine-learning jobs on PAI-Studio or PAI-DSW notebook. Relative detailed instructions are introduced in [TensorFlow manual](https://help.aliyun.com/document_detail/49571.html?spm=a2c4g.11186623.6.579.10501312JxztvO), you can also quickly jump to chapter "Quick Start".

FastNN Features：
* Models

    a.Some classic models on CV, including inception, resnet, mobilenet, vgg, alexnet, nasnet, etc;
    
    b.Preparing more state-of-art models on CV and Natural Language Processing (NLP), including Bert, XLnet, NMT, GPT-2, etc;

* Distributed Training (Turn to Alibaba PAI for submitting jobs on PAI-Studio or PAI-DSW notebook)

    a.Single-Node Multi-GPU
    
    b.Multi-Node Multi-GPU

* Half-Precision Training

* Task Type

    a.Model Pretrain
    
    b.Model Finetune：Only restore  trainable variables by default. Please turn to function 'get_assigment_map_from_checkpoint' in file "images/utils/misc_utils.py" for any other requirements.

We choose ResNet-v1-50 model and conduct large-scale test on Alibaba Cloud Computing Cluster(GPU P100). As the chart shows, PAISoar performs perfectly with near-linear scaling acceleration.

![resnet_v1_50](https://pai-online.oss-cn-shanghai.aliyuncs.com/fastnn-data/readme/resnet_v1_50_en.png)

## 2. Quick Start
This chaper is all about intructions on FastNN usage without any code modification, including：
* Data Preparing: Preparing local or PAI Web training data;
* Kick-off training: Setting for local shell script or PAI Web training parameters.

### 2.1 Data Preparing
For the convenience of trying  models in FastNN model Library, we prepare some open datasets (including cifar10, mnist and flowers) or their relative shell scripts for downloading and converting.

#### 2.1.1 Local datasets
Learning from TF-Slim model library, we provide  'images/datasets/download_and_convert_data.py' for downloading and converting to TFRecord format. Take cifar10 for example, script as following:
```
DATA_DIR=/tmp/data/cifar10
python download_and_convert_data.py \
	--dataset_name=cifar10 \
	--dataset_dir="${DATA_DIR}"
```
We will get the following tfrecord files in /tmp/data/cifar10 after running the script above:
```
$ ls ${DATA_DIR}
cifar10_train.tfrecord
cifar10_test.tfrecord
labels.txt
```

#### 2.1.2 OSS datasets
For the convenience of trying  FastNN model library on [Alibaba PAI](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf), we already download some datasets and convert them into TFRecord format, including cifar10, mnist, flowers. They can be accessed by oss api in PAI,  and their oss paths show as:

|dataset|num of classes|training set|test set|storage path|
| :-----: | :----: | :-----:| :----:| :---- |
| mnist   |  10    |  3320  | 350   | BeiJing：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/mnist/ ShangHai：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/mnist/
| cifar10 |  10    | 50000  |10000  | BeiJing：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/cifar10/ ShangHai：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/cifar10/
| flowers |  5     |60000   |10000  | BeiJing：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/flowers/ ShangHai：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/flowers/

### 2.2 Kick-off Training
The main file in FastNN is 'train_image_classifiers.py'. User parameters as well as  relative instructions are summarized in file 'flags.py'.
For more information, please turn to chapter 3. Among all params, the most common six params are listed as  following:
* task_type：String type. Clarifying whether 'pretrain' or 'finetune', 'pretrain' by default;
* enable_paisoar：Bool type.  True by default, when trying FastNN locally, should be set False;
* dataset_name：String type. Indicating training dataset, like files 'cifar10.py, flowers.py, mnist.py' in 'images/datasets', 'mock' by default;
* train_files：String type. Indicating names of all training files separated by comma, default None;
* dataset_dir：String type. Indicating training dataset directory, None by default;
* model_name：String type. Indicating model name, options among ['resnet_v1_50', 'vgg', 'inception'], for more information, check images/models;

Particularly, if task_type is 'finetune', model_dir and ckpt_file_name need also be specified, which indicates checkpoint dir and checkpoint file name respectively.

We provide instructions for FastNN model libraty on "Local Trial" and "PAI Trial" as following.

#### 2.2.1 Local Trial
For now, the initial version of FastNN  does not support PAISoar locally. If only for single-gpu training task, user param 'enable_paisoar' should set False, and software requirements lists:

|software|version|
| :-----: | :----: |
|python|>=2.7.6|
|TensorFlow|>=1.8|
|CUDA|>= 9.0|
|cuDNN| >= 7.0|

We take training task of Resnet-v1-50 on cifar10 for example to clarify local trial mannual.
##### 2.2.1.1 Pretrain Shell Script
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
##### 2.2.1.2 Finetune Shell Script
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

#### 2.2.2 PAI Trial
PAI supports several state-of-art frameworks, including TensorFlow(compatible with community version of 1.4 and 1.8), MXNet(0.9.5), Caffe(rc3). For TensorFlow users, PAI provides a built-in PAISoar component for distribution training. For mannual, please turn to [FastNN-On-PAI](https://help.aliyun.com/document_detail/139435.html).


## 3. User Parameters Intructions
Chapter 2.2 clarifies some most important parameters, While still many params stay unknown to users. FastNN model library integrates requirements of models with PAISoar component, and summarizes all params in file 'flags.py'(you can also define new params). All existing parameters could be divided into six parts:
* Dataset Option: Specific basic information about training dataset, such as 'dataset_dir' indicating training dataset directory;
* Dataset PreProcessing Option: Specific preprocessing func and params on dataset pipeline;
* Model Params Option: Specific model base params, including model_name, batch_size;
* Learning Rate Tuning: Specific learning rate and relative tuning params;
* Optimizer Option: Specific optimizer and relative tuning params;
* Logging Option: Specific params for logging;
* Performance Tuning: Specific half-precision and other relative tuning params.

### 3.1 Dataset Option
|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|dataset_name|string|Indicating dataset name, Value options include: 'mock', 'cifar10', 'mnist', 'flowers', we choose 'mock' for pseudo data by default.|
|dataset_dir|string|Indicating path to input dataset, we set None by default for pseudo data.|
|num_sample_per_epoch|integer|Total num of samples in training dataset, which helps for decay of learning rate commonly.|
|num_classes|integer|Classes of training dataset, default 100|
|train_files|string|String of name of all training files separated by comma, such as"0.tfrecord,1.tfrecord"|



### 3.2 Dataset Preprocessing Tuning
|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|preprocessing_name|string|Collaborating with 'model_name' to define preprocessing function, you can refer to file 'preprocessing_factory.py' in 'images/preprocessing' for value options, we set it be None by default.|
|shuffle_buffer_size|integer|shuffle buffer size of training dataset, default 1024|
|num_parallel_batches|integer|Product with batch_size specifies number of map_and_batch threads, we set it be 8 by default.|
|prefetch_buffer_size|integer|Numbert of batch data to be prefetched for dataset pipeline, we set it be 32 by default.|
|num_preprocessing_threads|integer|Number of preprocessing threads for dataset prefetching, we set it be 16 by default.|
|datasets_use_caching|bool|Cache the compressed input data in memory. This improves the data input performance, at the cost of additional memory|

### 3.3 Model Params Option
|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|task_type|string|Value options include 'pretrain' and 'finetune', we conduct model pretraining task by default, set it be 'pretrain'.|
|model_name|string|Indicating which model to be trained on, default inception_resnet_v2, you can refer to file 'model_factory.py' in 'images/models' for value options.|
|num_epochs|integer|Number of training epochs, default 100|
|weight_decay|float|The weight decay on the model weights, default 0.00004|
|max_gradient_norm|float|Clip gradient to this global norm, default None for clip-by-global-norm diabled|
|batch_size|integer|The number of samples processed every iteration for one device, default 32|
|model_dir|string|Dirtory of checkpoint for model finetuning.|
|ckpt_file_name|string|Collaborating with 'model_dir' to specify absolute path of checkpoint (pre-trained model: model_dir + model.ckpt).|

### 3.4 Learning Rate Tuning
|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|warmup_steps|integer|How many steps we inverse-decay learning. default 0.|
|warmup_scheme|string|How to warmup learning rates. Options include:'t2t' refers to Tensor2Tensor way, start with lr 100 times smaller,then exponentiate until the specified lr. default 't2t'|
|decay_scheme|string|How we decay learning rate. Options include:1、luong234: after 2/3 num train steps, we start halving the learning rate for 4 times before finishing;2、luong5: after 1/2 num train steps, we start halving the learning rate for 5 times before finishing;3、luong10: after 1/2 num train steps, we start halving the learning rate for 10 times before finishing.|
|learning_rate_decay_factor|float|Learning rate decay factor, default 0.94|
|learning_rate_decay_type|string|Indicating how the learning rate is decayed. One of ["fixed", "exponential", or "polynomial"], default exponential|
|learning_rate|float|Starting value for learning rate, default 0.01|
|end_learning_rate|float|Lower bound for learning rate when decay not disabled, default 0.0001|

### 3.5 Optimizer Option
|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|optimizer|string|the name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop", "adamweightdecay". Set it be "rmsprop" by default.|
|adadelta_rho|float|The decay rate for adadelta, default 0.95, specially for Adadelta|
|adagrad_initial_accumulator_value|float|Starting value for the AdaGrad accumulators, default 0.1, specially for Adagrada|
|adam_beta1|float|The exponential decay rate for the 1st moment estimates, default 0.9, specially for Adam|
|adam_beta2|float|The exponential decay rate for the 2nd moment estimates, default 0.999, specially for Adam|
|opt_epsilon|float|Epsilon term for the optimizer, default 1.0, specially for Adam|
|ftrl_learning_rate_power|float|The learning rate power, default -0.5, specially for Ftrl|
|ftrl_initial_accumulator_value|float|Starting value for the FTRL accumulators, default 0.1, specially for Ftrl|
|ftrl_l1|float|The FTRL l1 regularization strength, default 0.0, specially for Ftrl|
|ftrl_l2|float|The FTRL l2 regularization strength, default 0.0, specially for Ftrl|
|momentum|float|The momentum for the MomentumOptimizer, default 0.9, specially for Momentum|
|rmsprop_momentum|float|Momentum for the RMSPropOptimizer, default 0.9|
|rmsprop_decay|float|Decay term for RMSProp, default 0.9|

### 3.6 Logging Option
|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|stop_at_step|integer|Indicating number of training steps, default 100|
|log_loss_every_n_iters|integer|Frequency for outputing loss info, default 10|
|profile_every_n_iters|integer|Frequency for outputing timeline, default 0|
|profile_at_task|integer|Node index to output timeline, default 0|
|log_device_placement|bool|Indicating whether or not to log device placement, default False|
|print_model_statistics|bool|Indicating whether or not to print trainable variables info, default false|
|hooks|string|Indicating hooks for training, default "StopAtStepHook,ProfilerHook,LoggingTensorHook,CheckpointSaverHook"|

### 3.7 Performanse Tuning Option
|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|use_fp16|bool|Indicating whether to train with fp16, default True|
|loss_scale|float|Loss scale value for training, default 1.0|
|enable_paisoar|bool|Indicating whether or not to use pai soar, default True.|
|protocol|string|Default grpc.For rdma cluster, use grpc+verbs instead|

## 4. Self-defined Model Exploration
If existing models can't meet your requirements, we allow inheriting dataset／models／preprocessing api for self-defined exploration. Before that, you may need to understand overall code architecture of FastNN model library(taking 'images' models for example, whose main  file is 'train_image_classifiers.py'):

```
# Initialize some model in models for network_fn, and may carries param 'train_image_size'
    network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=FLAGS.num_classes,
            weight_decay=FLAGS.weight_decay,
            is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
# Initialize some preprocess_fn
    preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.model_name or FLAGS.preprocessing_name,
                is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
# According to dataset_name, choose right tfrecord for training dataset, and call preprocess_fn to parse training dataset
    dataset_iterator = dataset_factory.get_dataset_iterator(FLAGS.dataset_name,
                                                            train_image_size,
                                                            preprocessing_fn,
                                                            data_sources,
# Based on network_fn、dataset_iterator, define loss_fn
    def loss_fn():
    	with tf.device('/cpu:0'):
      		images, labels = dataset_iterator.get_next()
        logits, end_points = network_fn(images)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(logits, tf.float32), weights=1.0)
        if 'AuxLogits' in end_points:
          loss += tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(end_points['AuxLogits'], tf.float32), weights=0.4)
        return loss
# Call PAI-Soar API to wrapper loss_fn with original optimizer
    opt = paisoar.ReplicatedVarsOptimizer(optimizer, clip_norm=FLAGS.max_gradient_norm)
    loss = optimizer.compute_loss(loss_fn, loss_scale=FLAGS.loss_scale)
# Based on loss and opt, define training tensor 'train_op'
    train_op = opt.minimize(loss, global_step=global_step)
```

### 4.1 For New Dataset
**FastNN model library has supported direct access to dataset of tfrecord format**, and implements dataset pipeline based on TFRecordDataset for model training(utils/dataset_utils.py), helps to cover cost of sample preparing.
In addition, data-spliting is not finely implemented, FastNN requires:
* The number of training files can be divided by number of workers;
* The number of samples among training files is even.

If your dataset file is format of tfrecord, please reference files cifar10/mnist/flowers in 'datasets', take cifar10 for example:
* Assume key_to_features of cifar10 to be:
```
features={
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
}
```
* Create a new file 'cifar10.py' in 'datasets' and edit as following:
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
* Supplement 'dataset_map' in file 'datasets/dataset_factory.py'
```
from datasets import cifar10
datasets_map = {
    'cifar10': cifar10,
}
```
* When executing scripts, you need set 'dataset_name' to be 'cifar10' and 'train_files' to be 'cifar10_train.tfrecord' for model training.

If your datasets are some other formats, dataset pipeline may need to update.

### 4.2 For New Model
For exploration of new models, please turn to files in 'images/models':

* Model convergents normally;
* Input or output apis look like models in "models", Type or shape of input apis share with that of preprocessing and Type or shape of output apis share with that of labels;
* No other distributed settings.

Here, datasets and preprocessing can be reused. However, you gonna the following supplements:

* Supplement 'models_map' and 'arg_scopes_map' in file 'models/model_factory.py';
* Import your model into 'images/models'.

### 4.3 For New Dataset Preprocessing
For exploration of new dataset preprocessing_fn, please turn to files in 'images/preprocessing':

* Input or output apis look like preprocessing_fn in directory "preprocessing", Type or shape of input apis share with that of dataset and Type or shape of output apis share with that of model inputs;

you gonna the following supplements:

* Supplement 'preprocessing_fn_map' in file 'preprocessing_factory.py';

* Import yout preprocessing func file into directory 'images/preprocessing'.

### 4.4 For New Loss_fn
For 'images' in FastNN model library, as main file 'train_image_classifiers.py' shows, we implement loss_fn with 'tf.losses.sparse_softmax_cross_entropy'.

You can directly modify 'loss_fn' in 'train_image_classifiers.py'. However, when trying out distributed training jobs with PAISoar, 'loss_fn' returns loss only which is limited unchangable. Any other variables can be passed globally as accuracy returned in 'loss_fn' of 'train_image_classifiers.py'.

## 5. Acknowledgements
FastNN is an opensource project to indicate our contribution to distributed model library based on PAISoar. We believe FastNN allow researchers to most effectively explore various neural networks, and support faster data parallelism.
We will carry on with more innovation work on distribution, including model parallelism and gradient compression and so on.

FastNN for now includes only some state-of-art models on computer vision. Models on NLP (Bert, XLNet, GPT-2, NMT, etc)  or other fileds are comming soon.

FastNN now focuses on data parallelism, all models in 'images/models' are noted from [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim#tensorflow-slim-image-classification-model-library).
Thanks to TensorFlow community for implementation of these models. If any questions, please email us whenever you would like.
