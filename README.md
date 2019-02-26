# Retinal-Vessel-Segmentation
Retinal Vessel Segmentation based on Fully Convolutional Networks

Table 1: Segmentation results of different deep learning based methods on DRIVE. Bold values show the best score among all methods.

| Method               | F1         | Sn         | Sp         | Acc        | AUC        |
| -------------------- |:----------:|:----------:|:----------:|:----------:|:----------:|
| Melinscak et al. [1] | -          | 0.7276     | 0.9785     | 0.9466     | 0.9749     |
| Li et al. [2]        | -          | 0.7569     | 0.9816     | 0.9527     | 0.9738     |
| Liskowski et al. [3] | -          | 0.7520     | 0.9806     | 0.9515     | 0.9710     |
| Fu et al [4]         | -          | 0.7603     | -          | 0.9523     | -          |
| Oliveira et al. [5]  | -          | **0.8039** | 0.9804     | 0.9576     | **0.9821** |
| M2U-Net  [6]         | 0.8091     | -          | -          | **0.9630** | 0.9714     |
| R2U-Net [7]          | 0.8171     | 0.7792     | 0.9813     | 0.9556     | 0.9784     |
| LadderNet  [8]       | **0.8202** | 0.7856     | 0.9810     | 0.9561     | 0.9793     |
| **This work**        | 0.8169     | 0.7728     | **0.9826** | 0.9559     | 0.9794     |
