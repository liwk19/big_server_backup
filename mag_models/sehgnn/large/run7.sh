export CUDA_VISIBLE_DEVICES=5
python main.py --stages 400 400 400 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 200 --gama 10 --moving-k 1 --amp --seed 11 \
    --moe_widget ngnn1 feat_ngnn label_ngnn --num_expert 4 --top_k 1
# 这张卡给yufei

# conda env: drgat
