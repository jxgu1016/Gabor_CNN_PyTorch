# WACV2018: Gabor Convolutional Networks

Official PyTorch implementation of Gabor CNN. 
But all the results in the paper are based on [Torch 7](https://github.com/bczhangbczhang/Gabor-Convolutional-Networks).
These two implementations are sharing the same infrastructure level code.

## Install

```
git clone https://github.com/jxgu1016/Gabor_CNN_PyTorch
cd Gabor_CNN_PyTorch
sh install.sh
```

## Install third party tool
```
pip install tensorboardX
```

## Run MNIST demo

```
cd demo
python main.py --model gcn (--gpu 0)
```

## Please cite:
@article{Luan2016GCN, title={Gabor Convolutional Networks}, author={Luan, Shangzhen and Zhang, Baochang and Chen, Chen and Cao, Xianbin and Han, Jungong and Liu, Jianzhuang}, year={2017}, }
