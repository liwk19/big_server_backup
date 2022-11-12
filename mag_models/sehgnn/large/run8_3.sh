export CUDA_VISIBLE_DEVICES=6
python main.py --stages 400 400 400 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 3 \
    --moe_widget feat_ngnn label_ngnn --num_expert 8 --top_k 1 --reload de94d51b34c24d63b50c947d6474d7e5 --start-stage 1

# conda env: drgat
