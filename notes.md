
## Baseline 

```shell
python main.py --dataset cifar10 --arch vgg --depth 13 --batch-size 512 --epochs 99 --save ./vgg13/baseline
python main.py --dataset cifar10 --arch resnet --depth 164 --batch-size 256 --epochs 99 --save ./resnet164/baseline
python main.py --dataset cifar10 --arch densenet --depth 40
```

## Train with Sparsity

```shell
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 13 --batch-size 512 --epochs 99 --save ./vgg13/sparsity
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 164 --batch-size 256 --epochs 99 --save ./resnet164/sparsity
python main.py -sr --s 0.00001 --dataset cifar10 --arch densenet --depth 40
```

## Prune
Note: **If prune percent over 0.5, sometimes some layers will be all pruned and would be error**.
```shell
python vggprune.py --dataset cifar10 --depth 13 --percent 0.4 --model ./vgg13/sparsity/model_best.pth.tar --save ./vgg13/0.4
python resprune.py --dataset cifar10 --depth 164 --percent 0.4 --model ./resnet164/sparsity/model_best.pth.tar --save ./resnet164/0.4 --test-batch-size 128
python denseprune.py --dataset cifar10 --depth 40 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.


## Fine-tune

```shell
python main.py --refine ./vgg13/0.4/pruned.pth.tar --dataset cifar10 --arch vgg --depth 13 --epochs 99 --batch-size 512 --save ./vgg13/0.4/fine-tune-99
python main.py --refine ./resnet164/0.4/pruned.pth.tar --dataset cifar10 --arch resnet --depth 164 --epochs 99 --batch-size 256 --save ./resnet164/0.4/fine-tune-99
```

## Results

The results are fairly close to the original paper, whose results are produced by Torch. Note that due to different random seeds, there might be up to ~0.5%/1.5% fluctation on CIFAR-10/100 datasets in different runs, according to our experiences.

Note: 

1. **This is differ from official calculation in paper bacause the input of CIFAR-10 is 32 x 32, not 224 x 224 in ImageNet**.

2. **Parameters and FLOPs are counted by [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)**.

### CIFAR10
|  CIFAR10-Vgg-13  | Baseline |  Sparsity (1e-4)            | Prune (40%)           |       Fine-tune-99(40%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  89.82   |            89.71            |        17.60        |         90.08         |
|    Parameters     |  9.41M   |             9.41M            |       2.64M        |         2.64M         |
|    FLOPs          |  0.46G   |            0.46G           |         0.35G        |         0.35G         |

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



|  CIFAR100-Densenet-40  | Baseline |    Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) | Prune(60%)  | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  73.27   |          73.29            |        67.67        |         73.76         |   19.18       |     73.19     |
|    Parameters     |  1.10M  |            1.10M            |        0.71M        |         0.71M         |  0.50M       |     0.50M    |


mask-impl 的步骤是：

1. 稀疏化训练
2. 获得剪枝mask
3. 用mask对上一步剪枝后的权重进行fine-tune

这样只是将权重置零，不改变网络结构，并不会带来加速和模型文件大小的减小。mask-impl的意义是：即使一层全部被剪枝，也不会出错，因为只是权重被置0。