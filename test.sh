PYTHONPATH=. torchrun --nproc-per-node=1 seedvr/projects/inference_seedvr2_7b.py --video_path input --output_dir output --seed 42 --res_h 1920 --res_w 1080 --sp_size 1
