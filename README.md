## LMD: Faster image reconstruction with latent masking diffusion

### Evaluation

As a sanity check, run evaluation using our ImageNet **fine-tuned** models:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">fine-tuned checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>1b25e9</tt></td>
<td align="center"><tt>51f550</tt></td>
<td align="center"><tt>2541f2</tt></td>
</tr>
<tr><td align="left">reference ImageNet accuracy</td>
<td align="center">83.368</td>
<td align="center">85.964</td>
<td align="center">86.912</td>
</tr>
</tbody></table>

Evaluate ViT-Base in a single GPU (`${IMAGENET_DIR}` is a directory containing `{train, val}` sets of ImageNet):
```
python main_finetune.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
```

### Pre-training
To pre-train ViT-base (recommended default) with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:
```
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --
```
or conduct:
```
sh pretrain.sh
```


### Fine-tuning
To fine-tune with **multi-node distributed training**, run the following on 4 nodes with 8 GPUs each:
```
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
or conduct:
```
sh fine_tune.sh
```


