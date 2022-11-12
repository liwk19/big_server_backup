export CUDA_VISIBLE_DEVICES=7
python main.py --stages 600 600 600 600 --num-hops 2 --label-feats --extra-embedding complex \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 200 --gama 10 --moving-k 1 --amp --seed 5 \
    --moe_widget feat_ngnn label_ngnn --num_expert 4 --top_k 1

python main.py --stages 600 600 600 600 --num-hops 2 --label-feats --extra-embedding complex \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 200 --gama 10 --moving-k 1 --amp --seed 6 \
    --moe_widget feat_ngnn label_ngnn --num_expert 4 --top_k 1

python main.py --stages 600 600 600 600 --num-hops 2 --label-feats --extra-embedding complex \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 200 --gama 10 --moving-k 1 --amp --seed 7 \
    --moe_widget feat_ngnn label_ngnn --num_expert 4 --top_k 1

python main.py --stages 600 600 600 600 --num-hops 2 --label-feats --extra-embedding complex \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 200 --gama 10 --moving-k 1 --amp --seed 8 \
    --moe_widget feat_ngnn label_ngnn --num_expert 4 --top_k 1

python main.py --stages 600 600 600 600 --num-hops 2 --label-feats --extra-embedding complex \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 200 --gama 10 --moving-k 1 --amp --seed 9 \
    --moe_widget feat_ngnn label_ngnn --num_expert 4 --top_k 1
