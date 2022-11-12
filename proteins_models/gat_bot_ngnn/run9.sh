export CUDA_VISIBLE_DEVICES=7
python3 gat_bot_ngnn.py --n-runs 1 --gpu 0 --seed 6 --lr 0.001 --n-hidden 120 --log-every 20 \
    --num_expert 16 --top_k 1 --n-epochs 2000 --fmoe2 240 --dropout 0.25 --input-drop 0.2 \
    --label_warmup 2002 2002 --label_onehot --gate naive --moe_widget ngnn \
    --lr_patience 60 --lr_factor 0.85 --expert_drop 0.0 --attn_alpha 0.0 --ngnn_alpha 0.0 --match_n_epochs 1000

python3 gat_bot_ngnn.py --n-runs 1 --gpu 0 --seed 9 --lr 0.001 --n-hidden 120 --log-every 20 \
    --num_expert 16 --top_k 1 --n-epochs 2000 --fmoe2 240 --dropout 0.25 --input-drop 0.2 \
    --label_warmup 2002 2002 --label_onehot --gate naive --moe_widget ngnn \
    --lr_patience 60 --lr_factor 0.85 --expert_drop 0.0 --attn_alpha 0.0 --ngnn_alpha 0.0