
CUDA_VISIBLE_DEVICES=5,6 python run.py --g_encoder multi_att --dataset_name mams --learning_rate 5e-5  --max_hop 5 --dep_tag_type n_conn --softmax_first False --opn corr  #mams  85.05 84.29

CUDA_VISIBLE_DEVICES=6,7 python run.py --g_encoder multi_att --dataset_name laptop --learning_rate 1e-5 --max_hop 5 --dep_tag_type n_conn --softmax_first True --opn corr --num_heads 8  # laptop 81.52  77.68

CUDA_VISIBLE_DEVICES=1,3 python run.py --g_encoder multi_att --dataset_name rest --learning_rate 5e-5 --dep_tag_type composed --softmax_first False  # rest 87.86 82.41