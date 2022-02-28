# Bert training

This repo contains BERT training examples with EPL. The model code is based on https://github.com/google-research/bert .

## Training setup.

### Get pretrained bert base model.

```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

### Prepare dataset

```
mkdir data
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
```

## Distributed Bert training

### Data Parallelism

EPL can easily transform the local bert training program to a distributed one by adding a few lines of code.
```python
epl.init(epl.Config(config_json))
epl.set_default_strategy(epl.replicate(device_count=1))
```
You can refer to `run_squad_dp.py` for detailed implementation.

The following command launches a data parallelism program with two model replicas.
```
epl-launch --num_workers 2 --gpu_per_worker 1 scripts/train_bert_base_dp.sh
```

### Pipeline parallelism

To implement Bert pipeline parallelism, EPL only needs to change the annotation and configuration, as follows:
```python
config_json["pipeline.num_micro_batch"] = 4
epl.init(epl.Config(config_json))

# model annotation
epl.set_default_strategy(epl.replicate(1))
model_stage0()
epl.set_default_strategy(epl.replicate(1))
model_stage1()
```
You can refer to `run_squad_pipe.py` and `modeling.py` for detailed implementation.

The following command launches a pipeline parallelism program with two stages.

```
epl-launch --num_workers 1 --gpu_per_worker 2 scripts/train_bert_base_pipe.sh
```

## Evaluation
After training, you can perform the following commands to get the evaluation results.

```bash
SQUAD_DIR=data
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ${output_dir}/predictions.json
```
You are expected to get f1 ~= 88.0, exact_match ~= 79.8 after 2 epochs.

