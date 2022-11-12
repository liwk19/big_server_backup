export CUDA_VISIBLE_DEVICES=0
python main.py --data_root_dir ./dataset/ --pretrain_path ./dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
    --gpu 0 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.5 \
    --input-drop=0.35 --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 256 --save kd \
    --backbone drgat --mode teacher --n-runs 3 --seed 0 \
    --num_expert 4 --top_k 1 --gate naive --expert_drop 0.2 --moe_alpha 0.0 --moe_widget ngnn

# 这里在输入层也加NGNN

# python main.py --data_root_dir ./dataset/ --pretrain_path ./dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
#     --gpu 0 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.6 \
#     --input-drop=0.45 --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 256 --save kd \
#     --backbone drgat --alpha 0.95 --temp 0.7 --mode student --n-runs 2 --seed 0 --n-epochs 2000 \
#     --num_expert 4 --top_k 1 --gate naive --expert_drop 0.0 --moe_alpha 0.0 --moe_widget ngnn
