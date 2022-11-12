export CUDA_VISIBLE_DEVICES=0
python main.py --stages 300 300 300 400 --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 2 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload cfa5aafd29a746e0a7f716fac7215fd7 --start-stage 2

python main.py --stages 300 300 300 400 --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 3 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload 688cfa2f2c994d789c4fb5ea3888e0ea --start-stage 2

python main.py --stages 300 300 300 400 --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 4 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload c90087211132486c8a36636c95e9b6be --start-stage 2

python main.py --stages 300 300 300 400 --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 5 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload ce7ae5c25f9d4899a7ddf4886db015ff --start-stage 2

# 最后这个是timekiller，如果起床前还有空闲时间就跑
python main.py --stages 300 300 300 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 4 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload 08c7373325b54ea198a3c7147ca20d8e --start-stage 2

# conda env: drgat
