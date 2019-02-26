# Retinal-Vessel-Segmentation
Retinal Vessel Segmentation based on Fully Convolutional Networks

## Results

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

## Bibliography
[1] Melinˇsˇcak, M., Prentaˇsi, P., & Lonari, S. (2015, January). Retinal vessel segmentation using deep neural networks. In VISAPP 2015 (10th International Conference on Computer Vision Theory and Applications).
[2] Li, Q., Feng, B., Xie, L., Liang, P., Zhang, H., & Wang, T. (2016). A Cross-Modality Learning Approach for Vessel Segmentation in Retinal Images. IEEE Trans. Med. Imaging, 35(1), 109-118.
[3] Liskowski, P., & Krawiec, K. (2016). Segmenting retinal blood vessels with deep neural networks. IEEE transactions on medical imaging, 35(11), 2369-2380.
[4] Fu, H., Xu, Y., Lin, S., Wong, D. W. K., & Liu, J. (2016, October). Deepvessel: Retinal vessel segmentation via deep learning and conditional random field. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 132- 139). Springer, Cham.
[5] Oliveira, A. F. M., Pereira, S. R. M., & Silva, C. A. B. (2018). Retinal Vessel Segmenta- tion based on Fully Convolutional Neural Networks. Expert Systems with Applications.
[6] Laibacher, T., Weyde, T., & Jalali, S. (2018). M2U-Net: Effective and Efficient Retinal Vessel Segmentation for Resource-Constrained Environments. arXiv preprint arXiv:1811.07738.
[7] Alom, M. Z., Hasan, M., Yakopcic, C., Taha, T. M., & Asari, V. K. (2018). Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation. arXiv preprint arXiv:1802.06955.
[8] Zhuang, J. (2018). LadderNet: Multi-path networks based on U-Net for medical image segmentation. arXiv preprint arXiv:1810.07810.
