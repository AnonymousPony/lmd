python main_finetune.py \
    --eval \
    --resume ckpt/mae_finetuned_vit_base.pth \
    --model vit_base_patch16 \
    --batch_size 16 \
    --data_path /datasets