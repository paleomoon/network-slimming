import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

print(model)

# 计算需要剪枝的变量个数total
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0] # BN层weight就是γ，bias就是β，weight的shape[0]就是卷积层out_channels

# 确定剪枝的全局阈值
bn = torch.zeros(total) # total个0
index = 0
# 将γ的绝对值拷贝到bn中
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone() # γ的绝对值
        index += size

# 按照权值大小排序
y, i = torch.sort(bn)
thre_index = int(total * args.percent) # args.percent为稀疏度参数
# 确定要剪枝的阈值
thre = y[thre_index]

#********************************预剪枝*********************************#
pruned = 0
cfg = [] # 记录每个层要保留的通道数
cfg_mask = [] # 记录剪枝后每个BN层的通道mask图
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        # 要保留的通道标记Mask图
        mask = weight_copy.gt(thre).float().cuda() # touch.gt() greater than，weight_copy中大于thre为要保留的为1，否则要剪掉的为0
        # 剪枝掉的通道数个数
        pruned = pruned + mask.shape[0] - torch.sum(mask) # 总通道数-要保留的通道数=剪掉的通道数
        # 将需要剪掉的BN通道的参数直接置0
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)

# Make real prune
print(cfg)
newmodel = vgg(dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()]) # 计算参数量
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0 # BN层序号
# 定义原始模型和新模型的每一层保留通道索引的mask
start_mask = torch.ones(3) # 当前层剪枝前的mask,开始图像三通道都保留。如果前一层有剪枝，则前一层剪枝完后需要修改下一层的start_mask，这样层之间才能连续。
end_mask = cfg_mask[layer_id_in_cfg] # 当前层剪枝后的mask
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    # 对BN层和ConV层都要剪枝。定义的VGG模型每个ConV层后接BN
    if isinstance(m0, nn.BatchNorm2d):
        # np.squeeze 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        # np.argwhere(a) 返回非0的数组元组的索引，其中a是要索引数组的条件。
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # 如果维度是1，那么就新增一维，这是为了和BN层的weight的维度匹配
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        # 注意start_mask在end_mask的前一层，这个会在裁剪Conv2d的时候用到
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg] # end_mask现在是下一个BN层的mask
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) # 前一层的mask
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) # 后面的BN层剪枝后的mask
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        # 注意卷积核Tensor维度为[n, c, w, h]，VGG两个卷积层连接，下一层的输入维度n就等于当前层的c
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone() # 使与前面的层保持一致
        w1 = w1[idx1.tolist(), :, :, :].clone() # 这里有问题？？？
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        # 注意卷积核Tensor维度为[n, c, w, h]，两个卷积层连接，下一层的输入维度n就等于当前层的c
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

print(newmodel)
model = newmodel
test(model)

from count_flops import count
flops, params = count(model)
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')

