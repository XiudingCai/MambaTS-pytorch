export CUDA_VISIBLE_DEVICES=0

model_name=MambaTS

root_path_name=./dataset/ETT-small
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
seq_len=720


pred_len=96
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 5 \
    --d_layers 2 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 16 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
    --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.001

pred_len=192
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 2 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 16 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
    --train_epochs 10 --patience 3 --batch_size 16 --learning_rate 0.001

pred_len=336
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 2 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 32 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
    --train_epochs 10 --patience 3 --batch_size 32 --learning_rate 0.0005

pred_len=720
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 2 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 128 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
    --train_epochs 10 --patience 3 --batch_size 16 --learning_rate 0.0005
