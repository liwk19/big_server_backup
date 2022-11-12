export CUDA_VISIBLE_DEVICES=0
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max \
    --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2
# python test.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max \
#     --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2 \
#     --model_load_path revgnn_wide.pth  --valid_cluster_number 3 --num_evals 10

# 448 layers, 224 channels
# conda env: drgat
