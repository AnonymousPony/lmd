python submitit_finetune.py \
    --job_dir finetune_output_dir_piecewise \
    --nodes 1 \
    --ngpus 1 \
    --batch_size 64 \
    --model vit_base_patch4 \
    --finetune  /checkpoint.pth \
    --epochs  \
    --warmup_epochs 1 \
    --blr 1e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --use-adan --max-grad-norm 0 --opt-eps 1e-8 --opt-betas 0.98 0.92 0.99 \
    --data_path /datasets
    