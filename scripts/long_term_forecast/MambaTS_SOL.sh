export CUDA_VISIBLE_DEVICES=0

model_name=MambaTS

root_path_name=./dataset/Solar
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar
seq_len=720


for pred_len in 96 192 336 720
do
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
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
        --dropout 0.3 \
        --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
        --train_epochs 10 --patience 3 --batch_size 16 --learning_rate 0.0005
done
