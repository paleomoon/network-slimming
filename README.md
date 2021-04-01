# Network Slimming (Pytorch)

This repository contains an official pytorch implementation for the following paper  
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  
[Zhuang Liu](https://liuzhuang13.github.io/), [Jianguo Li](https://sites.google.com/site/leeplus/), [Zhiqiang Shen](http://zhiqiangshen.com/), [Gao Huang](http://www.cs.cornell.edu/~gaohuang/), [Shoumeng Yan](https://scholar.google.com/citations?user=f0BtDUQAAAAJ&hl=en), [Changshui Zhang](http://bigeye.au.tsinghua.edu.cn/english/Introduction.html).  

Original implementation: [slimming](https://github.com/liuzhuang13/slimming) in Torch.    
The code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming). We add support for ResNet and DenseNet.  

Citation:
```
@InProceedings{Liu_2017_ICCV,
    author = {Liu, Zhuang and Li, Jianguo and Shen, Zhiqiang and Huang, Gao and Yan, Shoumeng and Zhang, Changshui},
    title = {Learning Efficient Convolutional Networks Through Network Slimming},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
}
```


## Dependencies
torch v0.3.1, torchvision v0.2.0

## Channel Selection Layer
We introduce `channel selection` layer to help the  pruning of ResNet and DenseNet. This layer is easy to implement. It stores a parameter `indexes` which is initialized to an all-1 vector. During pruning, it will set some places to 0 which correspond to the pruned channels.

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg`,`resnet` or
`densenet`. The depth is chosen to be the same as the networks used in the paper.
```shell
python main.py --dataset cifar10 --arch vgg --depth 16 --batch-size 1024 --epochs 99 --save ./vgg16/baseline
python main.py --dataset cifar10 --arch resnet --depth 164 --batch-size 256 --epochs 99 --save ./resnet164/baseline
python main.py --dataset cifar10 --arch densenet --depth 40
```

## Train with Sparsity

```shell
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 16 --batch-size 1024 --epochs 99 --save ./vgg16/sparsity
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 164 --batch-size 256 --epochs 99 --save ./resnet164/sparsity
python main.py -sr --s 0.00001 --dataset cifar10 --arch densenet --depth 40
```

## Prune
Note: **If prune percent over 0.5, sometimes some layers will be all pruned and would be error**.
```shell
python vggprune.py --dataset cifar10 --depth 16 --percent 0.4 --model ./vgg16/sparsity/checkpoint.pth.tar --save ./vgg16/0.4
python resprune.py --dataset cifar10 --depth 164 --percent 0.4 --model ./resnet164/sparsity/model_best.pth.tar --save ./resnet164/0.4 --test-batch-size 128
python denseprune.py --dataset cifar10 --depth 40 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.

如果稀疏化训练保存模型之前使用了thop，在剪枝时会报错：Unexpected key(s) in state_dict: "total_ops", "total_params"。

3种解决方法：1. 在保存模型后使用thop。 2. model.load_state_dict(checkpoint['state_dict'],strict=False)。 3. 手动过滤掉"total_ops", "total_params"等参数。

参考：https://blog.csdn.net/daixiangzi/article/details/108368980 ， https://blog.csdn.net/qq_32998593/article/details/89343507



## Fine-tune

```shell
python main.py --refine ./vgg16/0.4/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16 --epochs 99 --batch-size 1024 --save ./vgg16/0.4/fine-tune-99
python main.py --refine ./resnet164/0.4/pruned.pth.tar --dataset cifar10 --arch resnet --depth 164 --epochs 99 --batch-size 256 --save ./resnet164/0.4/fine-tune-99
```

## Results

The results are fairly close to the original paper, whose results are produced by Torch. Note that due to different random seeds, there might be up to ~0.5%/1.5% fluctation on CIFAR-10/100 datasets in different runs, according to our experiences.

Note: 

1. **This is differ from official calculation in paper bacause the input of CIFAR-10 is 32 x 32, not 224 x 224 in ImageNet**.

2. **Parameters and FLOPs are counted by [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)**.

### CIFAR10
|  CIFAR10-Vgg-16  | Baseline |  Sparsity (1e-4)            | Prune (40%)           |       Fine-tune-99(40%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  89.00   |            88.00            |        10.00        |         89.32         |
|    Parameters     |  14.72M  |            14.72M            |        4.64M        |         4.64M         |
|    FLOPs          |  0.63G   |            0.63G           |         0.47G        |         0.47G         |

|  CIFAR10-Resnet-164  | Baseline |    Sparsity (1e-5)        | Prune(40%)      | Fine-tune-99(40%) |   Prune(60%)     |  Fine-tune-99(60%)       |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |  :----------------:| :--------------------:|
| Top1 Accuracy (%) |  89.62   |            88.70             |        48.70       |         89.47         |      14.20       |   89.12      |
|    Parameters     |  1.70M   |             1.70M            |        1.39M        |         1.39M         |      1.10M          |   1.10M           |
|    FLOPs          |  0.51G   |            0.51G             |         0.42G        |         0.42G         |      0.33G       |   0.33G           |

|  CIFAR10-Densenet-40  | Baseline |  Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) |       Prune(60%)   | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: | :--------------------: | :-----------------:|
| Top1 Accuracy (%) |  94.11   |           94.17             |        94.16       |         94.32         |      89.46       |     94.22     |
|    Parameters     |  1.07M  |            1.07M            |        0.69M       |         0.69M         |       0.49M      |    0.49M     |

### CIFAR100
|  CIFAR100-Vgg  | Baseline |   Sparsity (1e-4) | Prune (50%) | Fine-tune-160(50%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |   72.12   |            72.05             |         5.31        |         73.32         |
|    Parameters     |  20.04M  |            20.04M            |        4.93M        |         4.93M         |

|  CIFAR100-Resnet-164  | Baseline |   Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) |    Prune(60%)  | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  76.79   |            76.87             |        48.0        |         77.36        |  ---       |     ---     |
|    Parameters     |  1.73M  |            1.73M            |        1.49M        |         1.49M         |---       |     ---     |

Note: For results of pruning 60% of the channels for resnet164-cifar100, in this implementation, sometimes some layers are all pruned and there would be error. However, we also provide a [mask implementation](https://github.com/Eric-mingjie/network-slimming/tree/master/mask-impl) where we apply a mask to the scaling factor in BN layer. For mask implementaion, when pruning 60% of the channels in resnet164-cifar100, we can also train the pruned network.

|  CIFAR100-Densenet-40  | Baseline |    Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) | Prune(60%)  | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  73.27   |          73.29            |        67.67        |         73.76         |   19.18       |     73.19     |
|    Parameters     |  1.10M  |            1.10M            |        0.71M        |         0.71M         |  0.50M       |     0.50M    |

## Contact
sunmj15 at gmail.com 
liuzhuangthu at gmail.com  
