## MAE-SFER

MAE pre-training models (ViT-base, ViT-small, ViT-tiny) using 270K AffectNet images for static facial expression recognition (SFER).

## ViTs pre-trained on AffectNet
MAE ViT-Base pre-training on 270K AffectNet with a single 3090 GPU:

```
python -m torch.distributed.launch main_pretrain.py \
--model mae_vit_base_patch16 \
--batch_size 32 \
--accum_iter 4 --mask_ratio 0.75 \
--blr 1.5e-4 \
--epochs 300 \
--warmup_epochs 40 --weight_decay 0.05 \
--data_path /data/tao/fer/dataset/AffectNetdataset/Manually_Annotated_Images \
--output_dir /path/to/./out_dir_base
```
<details>
<summary>
ViT-Small and ViT-Tiny follow the same parameter settings as ViT-Base for MAE pre-training, except for the --model and --output_dir.
</summary>
--model mae_vit_small_patch16 and  --output_dir /path/to/./out_dir_small for ViT-Small

--model mae_vit_tiny_patch16 and  --output_dir /path/to/./out_dir_tiny for ViT-Tiny
</details>

ConvNeXt V2-Base pre-training on 270K AffectNet with a single 3090 GPU:

```
python -m torch.distributed.launch main_pretrain_convnextv2.py \
--model convnextv2_base \
--batch_size 64 --update_freq 8 \
--blr 1.5e-4 \
--epochs 400 \
--warmup_epochs 40 \
--data_path /data/tao/fer/dataset/AffectNetdataset/Manually_Annotated_Images \
--output_dir /path/to/./out_dir_base_1
```
## Fine-Tuning
MAE ViT-Base fine-tuning on 270K AffectNet with a single 3090 GPU:
```
python -m torch.distributed.launch main_finetune_affectnet.py \
--model mae_vit_base_patch16 \
--batch_size 16 \
--accum_iter 2 \
--blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 \
--drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
--epochs 100 \
--finetune '/path/out_dir_base_1/vit_base_checkpoint-299.pth'
```

> In addition, **data augmentation tricks** can significantly improve fine-tuning performance. Such as flip, colorjit, affine transformation, RandomErase, and mixup.


<details>
<summary>
ViT-Small and ViT-Tiny follow the same parameter settings as ViT-Base for MAE fine-tuning, except for the --model and --finetune.
</summary>
--model mae_vit_small_patch16 and --finetune /path/out_dir_small_1/vit_small_checkpoint-300.pth for ViT-Small

--model mae_vit_tiny_patch16 and --finetune /path/out_dir_tiny_1/vit_tiny_checkpoint-300.pth for ViT-Tiny

</details>



ConvNeXt V2-Base fine-tuning on RAF-DB with a single 3090 GPU:
```
python -m torch.distributed.launch main_finetune.py \
--model convnextv2_base \
--batch_size 32 --update_freq 4 \
--blr 6.25e-4 --epochs 100 --warmup_epochs 20 \
--layer_decay_type 'group' --layer_decay 0.6 --weight_decay 0.05 \
--drop_path 0.1 --reprob 0.25 \
--mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--data_path /path/to/Dataset/RAF-DB/basic \
--finetune '/path/out_dir_base/convnextv2_base_checkpoint-320.pth'
```

## Results and Pre-trained Models
### 270K AffectNet pre-trained weights for 300 epochs
| name | resolution | RAF-DB Acc(%) | AffectNet-7 Acc(%) | AffectNet-8 Acc(%) | FERPlus Acc(%) | #params | model |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE ViT-Base  | 224x224 | 91.07 | 66.09 | 62.42 | 90.18 | 86.5M | [model](https://drive.google.com/file/d/1mNruds0jDCkstYdH5VkHrkeoRqoRabgS/view?usp=drive_link) |
| MAE ViT-Small | 224x224 | 90.03 | 65.53 | 62.06 | 89.35 | 21.9M | [model](https://drive.google.com/file/d/1fPDoyHzrHwSKZI7dU7AcHd5dd2-ntwDk/view?usp=drive_link) |
| MAE ViT-Tiny  | 224x224 | 88.72 | 64.25 | 61.45 | 88.67 | 5.6M  | [model](https://drive.google.com/file/d/1wsXXVXlRP69RsbZiQD7GJUtCkI4JyJN7/view?usp=drive_link) |
| ConvNeXt V2-B | 224x224 | 89.52 |   -   |   -   |   -   | 89M   | [model](https://drive.google.com/file/d/1d56vRmpOu_r9Z8Fy61qxLyp_U6ZXP0YR/view?usp=drive_link) |

> Additional weights for ViT-Small（[600 epochs](https://drive.google.com/file/d/1keVIevQ2sirsOpZ4TE2dSehraUBuWLJl/view?usp=drive_link)） and ViT-Tiny([600](https://drive.google.com/file/d/1CZvTJRmswwMlX1Z5_48xJSMLdLtYAXq9/view?usp=drive_link) / [800](https://drive.google.com/file/d/11Zz4jFzN7fMqdp9XdJUe8gkVTOgBUUn1/view?usp=drive_link) / [1000](https://drive.google.com/file/d/1GwjtvvhffEtFNZWThWTTukf4Jap1QR5k/view?usp=drive_link) epochs) trained for more epochs are available.

> The accuracy of MAE ViT-Base increased to 91.79%, 63.81%, and 90.82% on RAF-DB, AffectNet-8, and FERPlus respectively with **data augmentation tricks**.

## Citation
If you find this repo helpful, please consider citing:

```
@article{li2024emotion,
  title={Emotion separation and recognition from a facial expression by generating the poker face with vision transformers},
  author={Li, Jia and Nie, Jiantao and Guo, Dan and Hong, Richang and Wang, Meng},
  journal={IEEE Transactions on Computational Social Systems},
  year={2024},
  publisher={IEEE}
}
```

```
@article{chen2024static,
  title={From static to dynamic: Adapting landmark-aware image models for facial expression recognition in videos},
  author={Chen, Yin and Li, Jia and Shan, Shiguang and Wang, Meng and Hong, Richang},
  journal={IEEE Transactions on Affective Computing},
  year={2024},
  publisher={IEEE}
}
```
