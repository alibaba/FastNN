## Hybrid Parallel(Whale) Tutorials
### 1. 简介

Whale提供统一、高效、简洁、易用的分布式训练框架，支持多种分布式训练模式：
- 数据并行
- 模型并行
- 流水并行
- 算子拆分
- 以上并行方式组合的混合并行

Hybrid Parallel Tutorials给出了多种分布式并行示例，包括以下目录：

- toy_examples：数据并行；
- bert：模型并行 + 流水并行；
- algo_with_split：算子拆分.


我们在Bert large等模型上进行了大规模的性能测试，在数据并行场景，Whale性能全面超过horovod，流水并行、算子拆分上性能又比数据并行有大幅提升。
- 在DLC集群(V100M16 + VPC 35Gb)上进行测试，训练和通信都使用float32；
- 从测试数据看，64 GPU卡时，Whale数据并行加速比比Horovod高**73.6%**；
- Whale流水并行加速比比Whale数据并行进一步提高**33.7%**.

### 2. 快速试用

本章节旨在给出模型库的使用说明. toy_examples目录下代码示例简单，下面仅对其余目录做详细说明。

模型库主文件为entry.py，其中训练脚本最常用的有以下六个参数：
- runner_name：字符串类型。取值为“images”、“bert”之一，指出训练的为images或bert目录下的某个模型；
- task_type：字符串类型。取值为“pretrain”、“finetune”之一，指出任务类型为模型预训练或模型调优；
- dataset_name：字符串类型。默认mock，指出训练数据解析文件，如images/datasets目录下的cifar10.py、flowers.py、mnist.py文件。
- dataset_dir：字符串类型。默认None，指出训练数据路径。
- model_name：字符串类型。默认inception_resnet_v2，指明模型名称，包括resnet_v1_50、vgg、inception等，详见images/models目录下的所有模型。
- average：正整数类型。指明whale.cluster划分策略为均分时每个slice的device数量，默认取值1。

特别地，当task_type=finetune时，需额外指定model_dir、ckpt_file_name参数，分别指明模型checkpoint路径及checkpoint文件名。

### 3. 用户参数指南
WhaleModelZoo库综合各个模型及Whale框架的需求，统一将可能用到的超参定义在shared_params.py文件（支持用户自定义新超参，可参考模型目录下的excluded_params.py文件）中，参数可分为以下六大类：
- whale超參：确定whale cluster切分方式;
- 数据集参数：确定训练集的基本属性，如训练集存储路径dataset_dir；
- 数据预处理参数：数据预处理函数及dataset pipeline相关参数；
- 模型参数：模型训练基本超參，包括model_name、batch_size等；
- 学习率参数：学习率及其相关调优参数；
- 优化器参数：优化器及其相关参数；
- 日志参数：关于输出日志的参数；
- 性能调优参数：混合精度等其他调优参数。

#### 3.1 whale超参
|    #名称              | #类型    | #描述            |
| :------------------: | :-----: | :-----------------  |
|    average           |integer   |指定whale.cluster一个slice的device数量。默认取值为1。|


#### 3.2 数据集

|    #名称              | #类型    | #描述            |
| :------------------: | :-----: | :-----------------  |
| dataset_name         | string  | 指定输入数据名称，默认mock_image |
| dataset_dir          | string  | 指定本地输入数据集路径，默认为None |
| file_pattern         | string  | 输出数据集文件后缀格式，默认为None |
| num_sample_per_epoch | integer | 数据集总样本数 |
| num_classes          | integer | 数据lable数，默认1001，对应imagenet数据分类数 |


#### 3.3 数据预处理

|    #名称              | #类型    | #描述            |
| :-----------------------: | :-----: | :-----------------  |
| preprocessing_name        | string  | 预处理方法名，默认None |
| shuffle_buffer_size       | integer | 数据shuffle的buffer size，默认1024 |
| num_parallel_batches      | integer | 与batch_size乘积为map_and_batch的并行线程数，默认8 |
| prefetch_buffer_size      | integer | 预取N个batch数据，默认N=32 |
| num_preprocessing_threads | integer | 预取线程数，默认为16 |
| datasets_use_caching      | bool    | Cache the compressed input data in memory. This improves the data input performance, at the cost of additional memory |

#### 3.4 模型基本参数

|    #名称      | #类型    | #描述            |
| :----------: | :-----: | :-----------------  |
| model_name   | string  | 指定模型名称，默认inception_resnet_v2 |
| loss_name    | string  | 指定loss函数，默认example |
| num_epochs   | integer | 训练epochs，默认100 |
| weight_decay | float   | The weight decay on the model weights, default 0.00004 |
| batch_size   | integer | The number of samples in each batch, default 32 |
| hooks        | string  | specify hooks for training, default "StopAtStepHook,ProfilerHook,LoggingTensorHook" |
| max_gradient_norm | float | clip gradients to this norm. default None.|


#### 3.5 学习率

|    #名称              | #类型    | #描述            |
| :------------------------: | :----: | :-----------------  |
| warmup_steps               | integer| how many steps we inverse-decay learning. |
| warmup_scheme              | string | how to warmup learning rates. Options include:'t2t' refers to Tensor2Tensor way, start with lr 100 times smaller,then exponentiate until the specified lr. default 't2t' |
| decay_scheme               | string | How we decay learning rate. Options include:luong234: after 2/3 num train steps, we start halving the learning rate for 4 times before finishing;luong5: after 1/2 num train steps, we start halving the learning rate for 5 times before finishing;luong10: after 1/2 num train steps, we start halving the learning rate for 10 times before finishing. |
| num_epochs_per_decay       | float  | number of epochs after which learning rate decays, default 2.0 |
| learning_rate_decay_factor | float  | learning rate decay factor, default 0.94 |
| learning_rate_decay_type   | string | specifies how the learning rate is decayed. One of ["fixed", "exponential", or "polynomial"], default exponential | 
| learning_rate              | float  | 学习率初始值，默认0.01 |
| end_learning_rate          | float  | decay时学习率值的下限 |

#### 3.6 optimizer

|    #名称              | #类型    | #描述            |
| :-------------------------------: | :----: | :-----------------  |
| optimizer                         | string | the name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop". Default "rmsprop" |
| adadelta_rho                      | float  | the decay rate for adadelta, default 0.95 |
| adagrad_initial_accumulator_value | float  | starting value for the AdaGrad accumulators, default 0.1 |
| adam_beta1                        | float  | the exponential decay rate for the 1st moment estimates, default 0.9 |
| adam_beta2                        | float  | the exponential decay rate for the 2nd moment estimates, default 0.999 |
| opt_epsilon                       | float  | epsilon term for the optimizer, default 1.0 |
| ftrl_learning_rate_power          | float  | the learning rate power, default -0.5 |
| ftrl_initial_accumulator_value    | float  | Starting value for the FTRL accumulators, default 0.1 |
| ftrl_l1                           | float  | The FTRL l1 regularization strength, default 0.0 |
| ftrl_l2                           | float  | The FTRL l2 regularization strength, default 0.0 |
| momentum                          | float  | The momentum for the MomentumOptimizer, default 0.9 |
| rmsprop_momentum                  | float  | Momentum for the RMSPropOptimizer, default 0.9 |
| rmsprop_decay                     | float  | Decay term for RMSProp, default 0.9 |

#### 3.7 Restore模型

|    #名称              | #类型    | #描述            |
| :--------------------: | :-----: | :-----------------  |
| model_dir      | string | dir of checkpoint for init |
| ckpt_file_name | string | Initial checkpoint (pre-trained model: base_dir + model.ckpt). |
| model_config_file_name  | string | The config json file corresponding to the pre-trained model. |
| vocab_file_name | string | The vocabulary file that the model was trained on.|

#### 3.8 日志参数

|    #名称              | #类型    | #描述            |
| :--------------------: | :-----: | :-----------------  |
| stop_at_step           | integer | the whole training steps, default 100 |
| log_loss_every_n_iters | integer | frequency to print loss info, default 10 |
| profile_every_n_iters  | integer | frequency to print timeline, default 0 |
| profile_at_task        | integer | node index to output timeline, default 0 |
| log_device_placement   | bool    | whether or not to log device placement, default False |
| print_model_statistics | bool    | whether or not to print trainable variables info, default false |
| hooks                  | string  | specify hooks for training, default "StopAtStepHook,ProfilerHook,LoggingTensorHook" |

#### 3.9 性能调优

|    #名称              | #类型    | #描述            |
| :----------------------: |  :---:  | :-----------------  |
| use_fp16                 |  bool   | whether to train with fp16, default True |
| loss_scale               |  float  | loss scale value for training, default 1.0 |
