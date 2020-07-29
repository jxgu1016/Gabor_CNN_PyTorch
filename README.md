# WACV2018/TIP: Gabor Convolutional Networks

Official PyTorch implementation of Gabor CNN. 
But all the results in the paper are based on [Torch 7](https://github.com/bczhangbczhang/Gabor-Convolutional-Networks).
These two implementations are sharing the same infrastructure level code.

## Requirements
- PyTorch 1.1.0 (earlier versions are not supported)
- torchvision
  
## Install

```
git clone https://github.com/jxgu1016/Gabor_CNN_PyTorch
cd Gabor_CNN_PyTorch
sh install.sh
```

## Run MNIST demo

```
cd demo
python main.py --model gcn (--gpu 0)
```

## Please cite:
@article{GaborCNNs, title={Gabor Convolutional Networks}, author={Luan, Shangzhen and chen, chen and Zhang, Baochang* and Han, jungong and Liu, Jianzhuang}, year={2018}, IEEE Trans. Image processing. }
