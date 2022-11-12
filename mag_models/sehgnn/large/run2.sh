export CUDA_VISIBLE_DEVICES=0
python main.py --stages 600 600 600 600 --num-hops 2 --label-feats --extra-embedding complex \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 200 --gama 10 --moving-k 1 --amp --seed 1 --hidden 256 \
    --moe_widget feat_ngnn label_ngnn --num_expert 4 --top_k 1
