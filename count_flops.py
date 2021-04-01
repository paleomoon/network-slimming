
import torch
from thop import profile

def count(model):
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model.cpu(), inputs=(input, ))
    return macs*2, params

# from ptflops import get_model_complexity_info
# macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))