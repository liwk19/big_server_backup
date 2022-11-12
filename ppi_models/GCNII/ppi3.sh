export CUDA_VISIBLE_DEVICES=1
python -u ppi.py --variant \
    --lr 0.001 --label_onehot --label_warmup 2000 2000 \
    --seed 0 --hidden 2048
    --epochs 12000
