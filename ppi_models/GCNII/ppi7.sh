export CUDA_VISIBLE_DEVICES=5
python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0003 --label_onehot --label_warmup 2000 2000 --moe_widget ngnn --expert_drop 0.1 \
    --seed 1 --hidden 1024 --moe_widget ngnn

python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0003 --label_onehot --label_warmup 2000 2000 --moe_widget ngnn --expert_drop 0.1 \
    --seed 2 --hidden 1024 --moe_widget ngnn

python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0003 --label_onehot --label_warmup 10000 10000 --moe_widget ngnn --expert_drop 0.1 \
    --seed 4 --hidden 1024 --moe_widget ngnn
