import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
import torch
import torch.nn.functional as F



class LSQWeightPruner(nn.Module):
    
    def __init__(self,n=4,m=8,quantizer=None):
        super(LSQWeightPruner, self).__init__()
        self.n=n
        self.m=m
        self.mask=None
        self.quantizer=quantizer
    def apply_n_m_sparse(self,weight, N, M):
        if weight.numel() % M != 0:
            raise ValueError("weight//M !!")
        
        reshaped_weights = weight.view(-1, M)
        _, idx = torch.topk(torch.abs(reshaped_weights), N, dim=1)
        mask = torch.zeros_like(reshaped_weights).scatter_(1, idx, 1)
        
        sparse_weights = reshaped_weights * mask
        return sparse_weights.view(weight.shape)
    def forward(self, weight):
        if self.quantizer  is not None:
            rst=self.quantizer(self.apply_n_m_sparse(weight,self.n,self.m))
            #print(rst.unique())
            #print(rst[0].cpu().detach().numpy())
            #1/0
            return rst
        return self.apply_n_m_sparse(weight,self.n,self.m)

class LSQActPruner(nn.Module):
    
    def __init__(self,n=4,m=8,quantizer=None):
        super(LSQActPruner, self).__init__()
        self.n=n
        self.m=m
        self.mask=None
        self.quantizer=quantizer
    def apply_n_m_sparse(self,x, N, M):
        if x.numel() % M != 0:
            raise ValueError("x//M !!")
        
        reshaped_weights = x.view(-1, M)
        _, idx = torch.topk(torch.abs(reshaped_weights), N, dim=1)
        mask = torch.zeros_like(reshaped_weights).scatter_(1, idx, 1)
        return mask.view(x.shape)
        # sparse_weights = reshaped_weights * mask
        # return sparse_weights.view(x.shape)
    def forward(self, x):
        if self.quantizer is not None and False:
            rst=self.quantizer(self.apply_n_m_sparse(x,self.n,self.m))
            return rst
        return self.apply_n_m_sparse(x.abs().mean(dim=(0,1)),self.n,self.m)

# # 示例：创建一个随机权重张量
# weights = torch.randn(3, 16,16)  # 假设这是一个4x4的权重矩阵

# # 应用2:4稀疏
# LP=LSQWeightPruner(4,8)
# sparse_weights = LP(weights)
# print("原始权重:\n", weights)
# print("稀疏化权重:\n", sparse_weights)       
