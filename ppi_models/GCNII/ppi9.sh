export CUDA_VISIBLE_DEVICES=7
python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 10000 10000 --expert_drop 0.2 \
    --seed 1 --hidden 2048 --moe_widget ngnn
