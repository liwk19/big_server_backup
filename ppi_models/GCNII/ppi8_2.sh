export CUDA_VISIBLE_DEVICES=6
python -u ppi.py --variant --num_expert 1 --top_k 1 \
    --lr 0.001 --label_onehot --label_warmup 2000 2000 --label_rate 0.3 --expert_drop 0.2 \
    --seed 0 --hidden 512 --moe_widget ngnn
