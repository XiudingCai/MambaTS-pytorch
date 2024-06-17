export CUDA_VISIBLE_DEVICES=0

model_name=MambaTS

root_path_name=./dataset/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom
seq_len=720


pred_len=96
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
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
    --d_model 32 \
    --d_ff 512 \
    --dropout 0.2 \
    --patch_len 48 --stride 48 \
    --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.0005 --VPT_mode 1 --ATSP_solver SA

pred_len=192
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
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
    --d_ff 512 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 \
    --train_epochs 10 --patience 3 --batch_size 16 --learning_rate 0.001 --VPT_mode 1 --ATSP_solver SA

pred_len=336
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
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
    --d_ff 512 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 \
    --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.001 --VPT_mode 1 --ATSP_solver SA

pred_len=720
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
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
    --d_ff 512 \
    --dropout 0.3 \
    --patch_len 48 --stride 48 \
    --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.001 --VPT_mode 1 --ATSP_solver SA
