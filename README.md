## MAE-SFER

MAE pre-training models (ViT-base, ViT-small, ViT-tiny) using 270K AffectNet images for static facial expression recognition (SFER).

## ViTs pre-trained on AffectNet
ConvNeXt V2-Base pre-training on 270K AffectNet with a single machine:

```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
--model convnextv2_base \
--batch_size 64 --update_freq 8 \
--blr 1.5e-4 \
--epochs 1600 \
--warmup_epochs 40 \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
Download the pre-trained weights(270K AffectNet)

1、MAE：[ViT-Base](https://drive.google.com/file/d/1mNruds0jDCkstYdH5VkHrkeoRqoRabgS/view?usp=drive_link)、[ViT-Small](https://drive.google.com/file/d/1fPDoyHzrHwSKZI7dU7AcHd5dd2-ntwDk/view?usp=drive_link)、[ViT-Tiny](https://drive.google.com/file/d/1wsXXVXlRP69RsbZiQD7GJUtCkI4JyJN7/view?usp=drive_link)

> Additional weights for ViT-Small（[600 epochs](https://drive.google.com/file/d/1keVIevQ2sirsOpZ4TE2dSehraUBuWLJl/view?usp=drive_link)） and ViT-Tiny([600](https://drive.google.com/file/d/1CZvTJRmswwMlX1Z5_48xJSMLdLtYAXq9/view?usp=drive_link) / [800](https://drive.google.com/file/d/11Zz4jFzN7fMqdp9XdJUe8gkVTOgBUUn1/view?usp=drive_link) / [1000](https://drive.google.com/file/d/1GwjtvvhffEtFNZWThWTTukf4Jap1QR5k/view?usp=drive_link) epochs) trained for more epochs are available.

2、Convnext V2：[ViT-Base](https://drive.google.com/file/d/1d56vRmpOu_r9Z8Fy61qxLyp_U6ZXP0YR/view?usp=drive_link)

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
