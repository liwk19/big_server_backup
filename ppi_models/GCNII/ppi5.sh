export CUDA_VISIBLE_DEVICES=3
python -u ppi.py --variant --num_expert 1 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 10000 10000 \
    --seed 2 --hidden 2048 --moe_widget ngnn --reload pretrained/edd5c8534d214158b6ff75c6cdd2a9bb.pt \
    --start_epoch 8000 --epochs 12000
