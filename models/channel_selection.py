import numpy as np
import torch
import torch.nn as nn

# 通道鉴别层放在每一个残差模块的第一个BN层后面以及整个网络的最后一个BN层后面，channel_selection前面的BN层不剪枝
# 目的是为了是保持残差模块输入输出通道以及整个网络输入输出通道不变，这样只在残差模块内部剪枝，方便很多
class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    从BN层的输出中选择通道。它应该直接放在BN层之后，此层的输出形状由self.indexes中的1的个数决定
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        使用长度和通道数相同的全1向量初始化"indexes", 剪枝过程中，将要剪枝的通道对应的indexes位置设为0
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        输入Tensor维度: (N,C,H,W)，这也是BN层的输出Tensor
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,)) 
        output = input_tensor[:, selected_index, :, :]
        return output