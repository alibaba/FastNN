
output_dir=$1
BERT_LARGE_DIR=uncased_L-24_H-1024_A-16
dataset=data
if [ -f "data/train.tf_record" ]; then
  cp data/train.tf_record $output_dir/
fi

python run_squad_pipe.py \
--vocab_file=$BERT_LARGE_DIR/vocab.txt \
--bert_config_file=$BERT_LARGE_DIR/bert_config.json \
--init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
--do_train=True \
--train_file=$dataset/train-v1.1.json \
--do_predict=True \
--predict_file=$dataset/dev-v1.1.json \
--train_batch_size=20 \
--predict_batch_size=1 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=384 \
--doc_stride=128 \
--output_dir=$output_dir \
--save_checkpoints_steps=2000 \
--version_2_with_negative=False \
--num_pipe_stages=2 \
--num_micro_batch=10 \
--profiler=False \
--gc=False \
--amp=False
