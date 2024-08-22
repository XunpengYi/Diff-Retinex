# [ICCV 2023] Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model
### [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yi_Diff-Retinex_Rethinking_Low-light_Image_Enhancement_with_A_Generative_Diffusion_Model_ICCV_2023_paper.pdf) | [Code](https://github.com/XunpengYi/Diff-Retinex)

This code is provided solely for academic research purposes and is exclusively restricted from any commercial use.

## 1. Create Environment
- Create Conda Environment
```
conda env create -f environment.yaml
```
- Activate Conda Environment
```
conda activate diff_retinex_env
```
We strongly recommend using the configurations provided in the yaml file, as different versions of dependency packages may produce varying results.

## 2. Prepare Your Dataset
You can refer to [LOL](https://daooshee.github.io/BMVC2018website/), [VE-LOL](https://flyywh.github.io/IJCV2021LowLight_VELOL/) to prepare your data. 

If you want to test only, you should list your dataset as the followed rule:
```bash
    dataset/
        your_dataset/
            train/
                high/
                low/
            eval/
                high/
                low/
```

## 3. Pretrain Weights
The pretrain weight for TDN is at [Google Drive](https://drive.google.com/drive/folders/1JmdvKUvzmCO1OoJJmNPTl8husw_pv1dn?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/151X72tHGrVrGuvRFt9goqQ) (code: cgbm). Please place it in the `model/Diff-TDN/weights` folder.

The pretrain weight for Diff-RDA is at [Google Drive](https://drive.google.com/drive/folders/1IAPafApa-FMDJ9CZL1aw1L1tUUkyHIVk?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/15_5yKuUd7uhGA7ZdhldVyg) (code: g6ln). Please place it in the `model/Diff-RDA/weights` folder.

The pretrain weight for Diff-IDA is at [Google Drive](https://drive.google.com/drive/folders/1r8kCmuYm3_ZscPb7l_tGlAO8-mMTTJ2t?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1pd79oR38b5ntBEaOSh3FeQ) (code: mikd). Please place it in the `model/Diff-IDA/weights` folder.
## 4. Testing
For low-light image enhancement testing, you need to modify the data path in the `config/Diff_Retinex_val.json` file, and then you can use the following command:
```shell
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py
```

Please note that recently many methods use the mean of the ground truth (GT-means) during testing, which may lead to better results. If you do not want to use this, you can disable it in test_from_dataset.py. We recommend ensuring consistent settings when making comparisons. (The results in `official_results` do not use GT-means.)

If the dataset is unpaired, please keep at least the same number and size of images in the `high` folder. It is normal for diffusion models to have some randomness, resulting in different enhancement outcomes.

**We found differences in the test results between machines equipped with RTX 3090 and RTX 4090. We recommend using only the RTX 3090 for testing to achieve the original performance. If you want to verify whether your results reflect the original performance, you can refer to the `official_results`.**

## 5. Train
If you need to train, you should train TDN, Diff-RDA, and Diff-IDA step by step. The training code is integrated into the `model` folder.

### Training the TDN
You need to modify the dataset path for training in `model/Diff_TDN/train_decom.py`, and then you can use the following command:
```shell
# For TDN training
python train_decom.py
```

### Training the Diff-RDA
Based on the weights trained for TDN, decompose the images from both the training set and the validation set, and organize the corresponding reflection maps dataset as follows:
```bash
    dataset/
        RDA_data/
            train/
                high/
                low/
            eval/
                high/
                low/
```
You need to modify the dataset path for training in `model/Diff_RDA/config` of Diff-RDA, and then you can use the following command:
```shell
# For Diff-RDA training
python train_rda.py
```

### Training the Diff-IDA
Based on the weights trained for TDN, decompose the images from both the training set and the validation set, and organize the corresponding illumination maps dataset as follows:
(Additionally, you need to place the ground truth images of normal light in the `gt/` folder and the reflectance maps restored from low light using Diff-RDA in the `R/` folder.)
```bash
    dataset/
        IDA_data/
            train/
                gt/
                R/
                high/
                low/
            eval/
                gt/
                R/
                high/
                low/
```
You need to modify the dataset path for training in `model/Diff_IDA/config` of Diff-IDA, and then you can use the following command:
```shell
# For Diff-IDA training
python train_ida.py
```
 

## Citation
If you find our work useful for your research, please cite our paper. 
```
@inproceedings{yi2023diff,
  title={Diff-retinex: Rethinking low-light image enhancement with a generative diffusion model},
  author={Yi, Xunpeng and Xu, Han and Zhang, Hao and Tang, Linfeng and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12302--12311},
  year={2023}
}
```
If you have any questions, please send an email to xpyi2008@163.com. 
