export CUDA_VISIBLE_DEVICES=4
python -u ppi.py --variant --num_expert 1 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 10000 10000 \
    --seed 3 --hidden 2048 --moe_widget ngnn

python -u ppi.py --variant --num_expert 1 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 10000 10000 \
    --seed 4 --hidden 2048 --moe_widget ngnn
