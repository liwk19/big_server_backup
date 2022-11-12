export CUDA_VISIBLE_DEVICES=5
python main.py --stages 300 300 300 300 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 4 \
    --moe_widget feat_ngnn --num_expert 4 --top_k 1

# 这个是在最后concat_project_layers之前加一个nn.Linear
# conda env: drgat
