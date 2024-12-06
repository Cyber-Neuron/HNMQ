import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
import numpy as np
converted = 0
seq_counter = 0
def hookmodel(model,conv,linear,pool=None,skipconv1=True,skip_last=False,others=None):
    global converted
    global seq_counter
    converted = 0
    seq_counter = 0
    # for k, m in enumerate(model.modules()):
    _t_hook(model._modules,conv,linear,pool, topn=1000,skipconv1=skipconv1,skip_last=skip_last,others=others)
def _t_hook(modules,conv,linear,pool, topn=1000, reverse=False,skipconv1=True,skip_last=False,others=None):
    global converted
    global seq_counter
    if reverse:
        mk = list(modules.items())[::-1]
    else:
        mk = modules.items()
    for k, (k_, m) in enumerate(mk):
        if hasattr(m, "_modules"):
            if len(m._modules) > 0:
                _t_hook(m._modules,conv,linear,pool, topn=topn, reverse=reverse,skipconv1=skipconv1,skip_last=skip_last,others=others)
        # print(others)
        # 1/0
        if others is not None: #replace a specific type of layer
            for T,S in others.items():
                # print(T,S,m.__class__.__name__,T.__class__.__name__)
                if isinstance(m,T):
                    # 1/0
                # if m.__class__.__name__==T.__class__.__name__:
                    # if "v_proj" in k_ or "k_proj" in k_ or True:# or "q_proj" in k_:
                    if "head" not in k_:
                        modules[k_] = S(m)
                #    print(k_)
        if isinstance(m, nn.Conv2d) and conv is not None:
            if m.in_channels == 3 and skipconv1: #  or m.kernel_size[0]==1:  # and False:
                continue
            else:
                if topn > 0:
                    if topn - converted > 0:
                        modules[k_] = conv.clone(m,conv)
                        converted += 1
                    else:
                        modules[k_] = m
                else:
                    modules[k_] = m 
        if isinstance(m, nn.Linear):
            if linear is not None:
                if skip_last and m.out_features==1000:
                    _=1
                else:
                    modules[k_] = linear.clone(m,linear)
        if isinstance(m,nn.AvgPool2d) and pool is not None:
            modules[k_]=pool #nn.AdaptiveAvgPool2d((1,1)) #for ttq baseline only
def quant_wrapper(model,conv,linear,pool,skip_first=True,skip_last=False,others=None):
    #if skip_last:
    #    linear=None
    hookmodel(model,conv,linear,pool,skipconv1=skip_first,skip_last=skip_last,others=others)
    return model

def set_bit(net,bit,key=None):
    if key is None:
        key="bit"
    for m in net.modules():
        if hasattr(m, key):
            setattr(m, key, bit)
def reg_quant(net,callback,key):
    for m in net.modules():
        if hasattr(m, key):
            # setattr(m, key, callback(in_c=m.lora_linear.base_layer.in_features,out_c=m.lora_linear.base_layer.out_features))
            setattr(m, key, callback(None,None))