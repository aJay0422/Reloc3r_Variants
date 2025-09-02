CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
torchrun --nproc_per_node=6 src/reloc3r_variants/train/train_Reloc3rWithDiffusionHead.py \
    --exp_name="reloc3r+diffusionhead-megadepth-6gpus-bs16 \
    --batch_size=16 \
    --accum_iter=2 \
    --epochs=100 \
    --lr 1e-5 \
    --min_lr 1e-7 \
    --warmup_epochs 5 \
    --world_size 6 \
    --eval_freq 3 \
    --save_freq 10 \
    --keep_freq 10 \
    --print_freq 20 \
    --output_dir checkpoints/reloc3r+diffusionhead-megadepth-6gpus-bs16