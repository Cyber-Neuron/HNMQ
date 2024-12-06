
import torch
import torch.nn as nn
from .util import quant_wrapper,reg_quant
from torch.nn import functional as F
from .quant_utils import Owraper
from torch.autograd import Function
from .pruner import LSQWeightPruner,LSQActPruner
import math
from typing import Any, Optional, Union
from .nlsq import LSQActivationQuantizer,LSQWeightQuantizer
from peft.tuners.lora import Linear as LoraLinear
class LTQ(nn.Module):
    def __init__(self, num_bits):
        super(LTQ, self).__init__()
        

    def forward(self, x):
        return x

class ClampPass(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        input=torch.clamp(input, -1, 1)

        return input


    @staticmethod
    def backward(ctx, grad_output) :

        input, =ctx.saved_tensors
        # grad_input = grad_output.clone()

        index = input.ge(1.0) + input.le(-1.0)

        grad_output[index]=0.
        return grad_output
def cLamppas(x):
    return ClampPass.apply(x)

class RoundPass(Function):
    @staticmethod
    def forward(ctx, input):
        x=torch. clamp(input, -1,1)
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def roundpass(x,t=0):
    return RoundPass.apply(x)

class normwt_msk(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, thre):
        xmask = x.abs().gt(thre.abs())  # upper bound
        #x=x.mul(xmask)
        x.mul_(xmask)
        w = F.normalize(x.sign().view(x.size(0), -1)).view(x.shape)
        nv2 = x * x
        nv2 = F.normalize(nv2.view(x.size(0), -1)).view(x.shape)
        s = (1 - nv2)  # /x_norm
        ctx.save_for_backward(s, xmask)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        s, xmask = ctx.saved_tensors
        grad = grad_output * s
        grad_thre = torch.mean(grad_output)
        return grad, grad_thre
class normwt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, thre):
        xmask = x.abs().gt(thre.abs())  # upper bound
        x=x.mul(xmask)
#        x.mul_(xmask)
        w = F.normalize(x.sign().view(x.size(0), -1)).view(x.shape)
        nv2 = w * w
        nv2 = F.normalize(nv2.view(x.size(0), -1)).view(x.shape)
        s = (1 - nv2)  # /x_norm
        ctx.save_for_backward(s, xmask)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        s, xmask = ctx.saved_tensors
        grad = grad_output * s
        grad_thre = torch.mean(grad_output)
        return grad, grad_thre
class wt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, thre):
        xmask = x.abs().gt(thre.abs())  # upper bound
        x=x.mul(xmask)
#        x.mul_(xmask)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output
        grad_thre = torch.mean(grad_output)
        return grad, grad_thre
def trace(m,x):
    
    if m.weight_quantizer is not None and m.training:
        weight=m.weight
        # print((m.weight.mul(10)).requires_grad,m.training)
        # 1/0
        mask=m.weight_quantizer(xref)
        qw=weight.view(m.weight.shape).mul(mask.view(mask.size(0),-1)).view(m.weight.size(0),-1)
        # weight.requires_grad_(True)
        # 1/0
        # qw=F.normalize(qw.view(m.weight.size(0),-1))
        if m.training and m.tt>0:
            
            m.weight_mat = F.cosine_similarity(m.weight,qw) #(1-torch.mm(weight, torch.t(qw)).diag())  ####
           
            m.weight_mat.requires_grad_(True)
            # print(m.weight_mat)
            # 1/0
                # m.weight_mat.sub_(torch.ones(m.weight.shape[0], device=m.weight_mat.device))#.mul_(m.scale_)
                
        else:
            
            weight=qw#.mul(weight_norm)
            m.weight_mat=None

        
    weight = weight.view(m.weight.shape) #+0.1*qw.view(m.weight.shape)  # F.normalize(wflat.view(self.weight.size(0), -1)).view(self.weight.shape)
 
    return weight





class HLinear(nn.Module):
    def __init__(self,lora_linear, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lora_linear=lora_linear
        self.weight_quantizer = None
        self.activation_quantizer = None
        self.tt=0.5
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        result=self.lora_linear(x, *args, **kwargs)
        if self.weight_quantizer is not None:
            result=trace(self,result)
        return result

class SLinear(nn.Linear):
    def __init__(self, linear,sparser=None) -> None:

        super().__init__(linear.in_features, linear.out_features, linear.bias is not None)
        self.sparser = sparser
        self.tt=0.5

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        weight=trace(self,x)
        # weight.requires_grad_(True)
        # print(weight)
        # 1/0
        out = F.linear(x, weight, self.bias)
        return out
    



def hyper_q_wrap(model, n=2,m=4,bits=4):
   
    model=quant_wrapper(model,None,None,None,others={nn.Linear:SLinear})
    
    # model=Owraper(model,sc=0.05)
    
    
    def wcallback(in_c,out_c):
        # return LSQActPruner(n=int(n),m=int(m))
        return LSQWeightPruner(n=int(n),m=int(m),quantizer=LSQWeightQuantizer(w_bits=int(bits),out_c=out_c,mean_scale=False,binary=bits==1,ternary=False))
    
    reg_quant(model, wcallback, "sparser")

    if False:
        for m in model.modules():
            if hasattr(m, "initialized_alpha"):
                m.initialized_alpha=True
 
    return model

