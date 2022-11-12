export CUDA_VISIBLE_DEVICES=1
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max \
    --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2
# python test.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max \
#     --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2 \
#     --model_load_path revgnn_deep.pth  --valid_cluster_number 3 --num_evals 10

# 1001 layers, 80 channels
# conda env: drgat
