export CUDA_VISIBLE_DEVICES=5
python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0003 --label_onehot --label_warmup 0 0 --label_rate 0.3 --expert_drop 0.0 \
    --seed 0 --hidden 1024 --moe_widget ngnn &> 19.txt

python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0003 --label_onehot --label_warmup 0 0 --label_rate 0.3 --expert_drop 0.1 \
    --seed 0 --hidden 1024 --moe_widget ngnn &> 20.txt
