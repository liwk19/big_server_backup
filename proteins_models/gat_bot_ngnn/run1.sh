export CUDA_VISIBLE_DEVICES=1

python3 visualize.py --gpu 0 --seed 2 --lr 0.001 --n-hidden 120 \
    --num_expert 16 --top_k 1 --match_n_epochs 2000 --fmoe2 240 --dropout 0.25 --input-drop 0.2 \
    --label_warmup 1400 1800 --label_onehot --gate sage --moe_widget ngnn \
    --lr_patience 60 --lr_factor 0.85 --features output
