export CUDA_VISIBLE_DEVICES=0

model_name=MambaTS

root_path_name=./dataset/traffic
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
seq_len=720


for pred_len in 720 336 192 96
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --num_workers 0 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 2 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 512 \
    --d_ff 512 \
    --dropout 0.2 \
    --patch_len $seq_len --stride $seq_len \
    --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.0005 --VPT_mode 1 --ATSP_solver SA
done
