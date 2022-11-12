export CUDA_VISIBLE_DEVICES=2
python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 2000 2000 \
    --seed 2 --hidden 2048 --moe_widget ngnn
    --epochs 12000
