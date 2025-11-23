# FDRect : Flow-Distilled Image Rectangling via Asymmetric Progressive Knowledge Transfer
<p align="center">KaiJun Wu<sup>1</sup>, XiaoRong Xu<sup>2 *</sup>, DanDan Feng<sup>3</sup>, Yuan Mei<sup>4</sup>, ChongKai Zhu<sup>5</sup></p>
<p align="center"><sup>1</sup>Lanzhou Jiaotong University,</p>
<p align="center"><sup>2</sup>Hong Kong Polytechnic University</p>



<div align=center>
<img src="https://github.com/XiaoRong24/FDRect/blob/main/Network.png"/>
</div>


## Requirement
* python 3.10.16
* numpy 1.24.4
* pytorch 2.5.0
* torchvision 0.20.0 

If you encounter some problems about the mamba environment, please refer to this [course](https://blog.csdn.net/weixin_51949030/article/details/144729352).

## Training
### Step1: Training the Aligment Model
Put flow knowledge into 'DIR-D/training/' folder
### Step2: Training the FDRect Model
```
cd ./code/
python train_flow.py
```

## Testing
Our pretrained rectangling model can be available at [Google Drive](https://drive.google.com/drive/folders/1YLYx1qpJqPf8ffW21hSeKGxiJO5ss5l-?usp=sharing). And place the two files to 'code/model/' folder.

```
cd code/
python test.py
```

## Dataset (DIR-D)
Source:[DeepRectangling](https://github.com/nie-lang/DeepRectangling)


## Meta
If you have any questions about this project, please feel free to drop me an email.

XiaoRong Xu -- 1084592611@qq.com
