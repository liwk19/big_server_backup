export CUDA_VISIBLE_DEVICES=0
python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 15000 15000 --expert_drop 0.2 \
    --seed 0 --hidden 2048 --moe_widget ngnn --reload pretrained/00570e7644b64fe6bfd86789a2120b2f.pt \
    --start_epoch 8000 --epochs 12000

python -u ppi.py --variant --num_expert 1 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 15000 15000 \
    --seed 2 --hidden 2048 --moe_widget ngnn --reload pretrained/edd5c8534d214158b6ff75c6cdd2a9bb.pt \
    --start_epoch 8000 --epochs 12000

python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 15000 15000 --expert_drop 0.2 \
    --seed 4 --hidden 2048 --moe_widget ngnn --reload pretrained/ab8df935fa204487b719ae55e948504f.pt \
    --start_epoch 8000 --epochs 12000

# python -u ppi.py --variant --num_expert 4 --top_k 1 \
#     --lr 0.0001 --label_onehot --label_warmup 15000 15000 --expert_drop 0.2 \
#     --seed 1 --hidden 2048 --moe_widget ngnn --reload pretrained/ab8df935fa204487b719ae55e948504f.pt \
#     --start_epoch 8000 --epochs 12000
