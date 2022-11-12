export CUDA_VISIBLE_DEVICES=5
python main.py --stages 300 300 300 400 --extra-embedding complex --num-hops 2 --label-feats \
    --num-label-hops 2 --n-layers-1 2 --n-layers-2 2 --residual --act leaky_relu --bns --label-bns \
    --lr 0.001 --weight-decay 0 --threshold 0.75 --patience 100 --gama 10 --moving-k 1 --amp --seed 2 \
    --moe_widget feat_ngnn --num_expert 8 --top_k 1 --reload 7354cb5705694b91af59d68c109bc9db --start-stage 3

# conda env: drgat
