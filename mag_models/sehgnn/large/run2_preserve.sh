export CUDA_VISIBLE_DEVICES=0
python main.py --stages 300 300 300 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 1 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload 8cbdc750f7864e4780fb2e5352d9f201 --start-stage 3

python main.py --stages 300 300 300 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 3 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload f6847b6136be4a1fb2405195436d44bc --start-stage 3

python main.py --stages 300 300 300 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 4 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload 08c7373325b54ea198a3c7147ca20d8e --start-stage 2

python main.py --stages 300 300 300 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 5 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload 1395c7d32f49461a81650a072e446f82 --start-stage 2

python main.py --stages 300 300 300 400 --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 1 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload ef794e2d96414cd69248744b55f8dd36 --start-stage 2

# conda env: drgat
