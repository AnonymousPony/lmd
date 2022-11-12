python submitit_pretrain.py \
    --job_dir output_dir \
    --nodes 1 \
    --ngpus 1 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_base_patch4 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs  \
    --warmup_epochs 1 \
    --blr 3e-4 --weight_decay 0.05 --min_lr 1e-8 \
    --data_path /datasets \
    --use-adan --opt-betas 0.98 0.92 0.90 --opt-eps 1e-8 --max-grad-norm 10.0 

#  --data_path /data/yzh/imagenet/datasets