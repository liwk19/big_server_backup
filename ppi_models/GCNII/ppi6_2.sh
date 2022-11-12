export CUDA_VISIBLE_DEVICES=4
python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0002 --label_onehot --label_warmup 0 0 --label_rate 0.3 --expert_drop 0.2 \
    --seed 0 --hidden 1024 --moe_widget ngnn &> 10.txt

python -u ppi.py --variant --num_expert 4 --top_k 1 \
    --lr 0.0001 --label_onehot --label_warmup 0 0 --label_rate 0.3 --expert_drop 0.2 \
    --seed 1 --hidden 1024 --moe_widget ngnn &> 11.txt
