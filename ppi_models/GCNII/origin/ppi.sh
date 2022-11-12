export CUDA_VISIBLE_DEVICES=2
python -u ppi.py --variant --test --hidden 2048 --seed 0 &> 1.txt
python -u ppi.py --variant --test --hidden 2048 --seed 1 &> 2.txt
python -u ppi.py --variant --test --hidden 2048 --seed 2 &> 3.txt
