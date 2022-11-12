export CUDA_VISIBLE_DEVICES=5
python -u gnn.py --runs 1 --data_root_dir ./dataset/ --use_sage --lr 8e-4 \
    --node_emb_path ../drgat/dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy

# conda env: drgat
