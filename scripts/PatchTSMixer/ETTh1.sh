if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/forecasting" ]; then
    mkdir ./logs/forecasting
fi


seq_len=336
model_name=PatchTSMixer
pred_len=96

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data=ETTh1
channels=7

python -u run_exp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $channels \
    --des 'Exp' \
    --train_epochs 2 \
    --itr 1 --batch_size 128 >logs/forecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$channel_independence.log 