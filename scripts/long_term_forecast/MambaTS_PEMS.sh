export CUDA_VISIBLE_DEVICES=0

model_name=MambaTS

root_path_name=./dataset/PEMS
data_path_name=PEMS03.npz
model_id_name=PEMS
data_name=PEMS
seq_len=720


model_name=MambaTSv3
for pred_len in 96 48 24 12
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 6 \
    --d_layers 2 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 512 \
    --dropout 0.2 \
    --patch_len 48 --stride 48 \
    --train_epochs 10 --patience 3 --batch_size 16 --learning_rate 0.001 --VPT_mode 1 --ATSP_solver SA
done
