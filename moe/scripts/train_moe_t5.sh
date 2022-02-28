output_dir=$1
data_dir=data/

problem=translate_ende_wmt32k
model=moe_transformer
hparams=moe_t5_small


python trainer.py \
  --data_dir=$data_dir \
  --problem=$problem \
  --model=$model \
  --dbgprofile=False \
  --hparams_set=$hparams \
  --output_dir=$output_dir \
  --train_steps=20 \
  --log_step_count_steps=1 \
  --schedule=train

