# ResNet training

This repo contains ResNet training examples with EPL.


## Distributed ResNet training

### Data Parallelism

EPL can easily transform the local bert training program to a distributed one by adding a few lines of code.
```python
epl.init(epl.Config(config_json))
epl.set_default_strategy(epl.replicate(device_count=1))
```
You can refer to `resnet_dp.py` for detailed implementation.

The following command launches a data parallelism program with two model replicas.
```
epl-launch --num_workers 2 --gpu_per_worker 1 scripts/train_dp.sh
```

### Tensor model parallelism

To implement ResNet tensor model parallelism,
EPL only needs to change the annotation and configuration, as follows:
```python
config = epl.Config({"cluster.colocate_split_and_replicate": True})
epl.init(config)

with epl.replicate(total_gpu_num):
  feature_extraction()

with epl.split(total_gpu_num):
  classification()
```
You can refer to `resnet_split.py` for detailed implementation.

The following command launches a tensor model parallelism program with two workers.

```
epl-launch --num_workers 2 --gpu_per_worker 1 scripts/train_split.sh
```


