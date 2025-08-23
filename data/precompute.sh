python data/fourier_vec.py --dataset_dir '/fs/nexus-scratch/haowenyu/SportsActionUnderstanding/ActionData' \
--cache_root '/fs/nexus-scratch/haowenyu/ActionDataCache' \
--mode 'fast' \
--accum_interval 150 \
--max_events 100000 \
--downsample_rate 0.1 \
--enc_dim 128 \
--use_cache \
--num_workers 4