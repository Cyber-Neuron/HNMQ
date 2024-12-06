import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad
    

def round_pass(x):
    # return Round.apply(x)
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad
def sign_pass(x):
    #return Round.apply(x)
    y = x.sign()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def quantize(x, step_size, low, high,scale=None,means=False,mu=torch.tensor(0.),sigma=torch.tensor(1.),grid=None,weight=False,binary=False) :
 
    scale=torch.tensor(1/(x.numel() * high) **0.5)
    # print(low,high,scale)
    # 1/0
    step_size = grad_scale(step_size, scale)
    if means:
        index = 0 #ttq.apply(x,step_size)
    else:
        # print(x.shape,step_size.size())
        # 1/0
        index = x / step_size
    if not binary:
        index = index.clamp(low, high) 
    if weight or True:
        if binary:
            # print(x,index,step_size)
            index = sign_pass(index)
            # print(index)
            # 1/0
        else:
            # print(x,index,step_size)
            index = round_pass(index)#+0.49999)-0.49999
            # print(index)
            # 1/0
    else:
        index=power_round(index,grid,mu,sigma)
    # print(scale)
    # 1/0
    # if step_size<0.9:
    #print(index.unique(),step_size,(index * step_size).unique())
    #1/0
    # print(index.shape,low,high,index.unique())

    return index * step_size
class Round(Function):
       
    @staticmethod
    def forward(self,x):
        xr = torch.round(x)
        nv2 = xr * xr
        nv2 = F.normalize(nv2.view(x.size(0), -1)).view(x.shape)
        s = (1 - nv2)  # /x_norm
        self.save_for_backward(s)
        return xr
    @staticmethod
    def backward(self, grad_output):
        '''
        i = (input.abs()>1.).float()
        sign = input.sign()
        grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
        
        '''
        
        s,= self.saved_tensors
    #    x=x-0.5
    #    df=x.round()-x
     #   grad_scale_elem=F.sigmoid(df)*(1-F.sigmoid(df))
        grad = grad_output * s#grad_scale_elem*4
        return grad
class lsq(Function):
    @staticmethod
    def forward(ctx, input,step_size,valmin,valmax,ags):
        rt=input/step_size#.abs()
        # print(valmin,valmax)
        x_clip=torch.clamp(rt,valmin,valmax)
        ctx.save_for_backward(x_clip,valmin,valmax,ags)
        x_round=torch.round(x_clip) #-0.5)+0.5
        x_restore = torch.mul(x_round, step_size)
        return x_restore

    @staticmethod
    def backward(ctx, grad_output):
        grad_top =grad_output
        x_clip,valmin,valmax,ags = ctx.saved_tensors
        internal_flag = ((x_clip > valmin).float() - (x_clip >= valmax).float()) #00000_valmin_111111_valmax_000000
        # internal_flag_r = ((x_clip > valmin).float() - 2*(x_clip >= valmax).float())
        # gradient for activation
        grad_activation = grad_top * internal_flag
        # x_clip=x_clip+0.5 
        # gradient for scale
        grad_one = x_clip * internal_flag #00000_ggggg_00000
        grad_two = torch.round(x_clip)    #00000_rrrrr_33333
        grad_scale_elem = grad_two - grad_one
        # grad_scale_elem_r =(grad_one-grad_two)*internal_flag_r
        # grad_scale_elem=(F.sigmoid(-1*x_clip)*torch.cos(torch.pi*x_clip)).abs()
        # grad_scale_elem=F.sigmoid(-1*x_clip)*grad_scale_elem.abs() #best
        if False:
            grad_scale_elem=F.sigmoid(grad_scale_elem)*(1-F.sigmoid(grad_scale_elem))#*grad_scale_elem.abs() #best
        
        # grad_scale_elem=grad_scale_elem.abs()*0.6
        # grad_scale_elem=grad_scale_elem.abs()
        if False: 
            nv2=x_clip*x_clip
            nv2=F.normalize(nv2.view(x_clip.size(0), -1)).view(x_clip.shape)
            sss=1-nv2
        else:
            sss=1 
        # grad_scale = (grad_scale_elem_s.abs() * grad_top*sss).sum().view((1,))
        # grad_scale = ((grad_scale_elem.pow(2)-0.25).abs() * grad_top*sss).sum().view((1,))
        grad_scale = (grad_scale_elem * grad_top*sss).sum().view((1,))
        #todo 试试挪0.5, 试试调整振幅(增大)
        # print(grad_scale_s.min(),grad_scale_s.max(),grad_scale_s.mean(),grad_scale_s.var())
        # print(grad_scale.min(),grad_scale.max(),grad_scale.mean(),grad_scale.var())
        # 1/0
        return grad_activation, grad_scale,None,None,None
class lsqw(Function):                                                                                                      
    @staticmethod                                                                                                          
    def forward(ctx, input,step_size,valmin,valmax,ags):                                                                   
        rt=input/step_size                                                                                          
        # print(valmin,valmax)                                                                                             
        x_clip=torch.clamp(rt,valmin,valmax)                                                                               
        ctx.save_for_backward(x_clip,valmin,valmax,ags)                                                                    
        x_round=torch.round(x_clip)                                                                                        
        x_restore = torch.mul(x_round, step_size)                                                                    
        return x_restore                                                                                                   
                                                                                                                           
    @staticmethod                                                                                                          
    def backward(ctx, grad_output):                                                                                        
        grad_top =grad_output                                                                                              
        x_clip,valmin,valmax,ags = ctx.saved_tensors                                                                       
        internal_flag = ((x_clip>valmin).float() - (x_clip >= valmax).float()) #00000_valmin_111111_valmax_000000        
        # internal_flag_r = ((x_clip valmin).float() - 2*(x_clip >= valmax).float())                                     
        # gradient for activation                                                                                          
        grad_activation = grad_top #* internal_flag                                                                        
        # x_clip=x_clip+0.5                                                                                                
        # gradient for scale                                                                                               
        grad_one = x_clip * internal_flag #00000_ggggg_00000                                                               
        grad_two = torch.round(x_clip)    #00000_rrrrr_33333                                                               
        grad_scale_elem = grad_two - grad_one                                                                              
        # grad_scale_elem_r =(grad_one-grad_two)*internal_flag_r                                                           
        # grad_scale_elem=(F.sigmoid(-1*x_clip)*torch.cos(torch.pi*x_clip)).abs()                                          
        # grad_scale_elem=F.sigmoid(-1*x_clip)*grad_scale_elem.abs() #best                                                 
        if False:                                                                                                          
            grad_scale_elem=F.sigmoid(grad_scale_elem)*(1-F.sigmoid(grad_scale_elem))#*grad_scale_elem.abs() #best         
                                                                                                                           
        # grad_scale_elem=grad_scale_elem.abs()*0.6                                                                        
        # grad_scale_elem=grad_scale_elem.abs()
        if False: 
            nv2=x_clip*x_clip 
            nv2=F.normalize(nv2.view(x_clip.size(0), -1)).view(x_clip.shape)
            sss=1-nv2
        else:
            sss=1 
        # grad_scale = (grad_scale_elem_s.abs() * grad_top*sss).sum().view((1,))
        # grad_scale = ((grad_scale_elem.pow(2)-0.25).abs() * grad_top*sss).sum().view((1,))
        grad_scale = (grad_scale_elem * grad_top*sss).mean().view((1,))
        #todo 试试挪0.5, 试试调整振幅(增大)
        # print(grad_scale_s.min(),grad_scale_s.max(),grad_scale_s.mean(),grad_scale_s.var())
        # print(grad_scale.min(),grad_scale.max(),grad_scale.mean(),grad_scale.var())
        # 1/0
        return grad_activation, grad_scale,None,None,None

        #return quantize(x,self.w_step_size,low=self.wlow.to(x.device),high=self.whigh.to(x.device),means=self.mean_scale,weight=True)
#        return lsqw.apply(x,self.w_step_size,self.wlow.to(x.device),self.whigh.to(x.device),self.ags)


class LSQActivationQuantizer(nn.Module):
    def __init__(self, a_bits,mean_scale,binary=False,ternary=True):
        super(LSQActivationQuantizer, self).__init__()
        self.a_bits=a_bits
        self.act_step_size =nn.Parameter(torch.tensor(1.0)) #torch.tensor(1.0)# 
        # self.act_step_size_tmp =nn.Parameter(torch.tensor(2.0)) #torch.tensor(1.0)# 
        if binary:
            self.ahigh=torch.tensor(1.0)   
        elif ternary: 
            self.ahigh=torch.tensor(2 ** self.a_bits - 2) #ternary
        else:
            self.ahigh=torch.tensor(2 ** self.a_bits - 1) #n-bit
        # self.ahigh=torch.tensor(2 **(self.w_bits-1)-1)
        # self.alow=torch.tensor(-2 **(self.w_bits - 1))
        #self.wlow=torch.tensor(-2 **(self.w_bits - 1))
        #self.whigh=torch.tensor(2 **(self.w_bits-1)-1)
        self.mean_scale=mean_scale
        self.ags=torch.tensor(0)
        self.init_batchs=0
        # self.mu = nn.Parameter(torch.zeros(4))  # Mean of Z
        # self.sigma = nn.Parameter(torch.ones(4)) # Standard deviation of Z
        # self.grid=torch.tensor([0,1,2,3])
    # def reparameterize(self, epsilon):
    #     return self.mu + self.sigma * epsilon
    def forward(self, x):
        # epsilon = torch.randn_like(x)*0.01
        # z = self.reparameterize(epsilon)
        if False and self.act_step_size.abs()<0.9:
            self.act_step_size.data=self.act_step_size.sign()*torch.tensor(0.9).to(self.act_step_size.device)
        # return quantize(x,self.act_step_size,low=0,high=self.ahigh,means=self.mean_scale)#,mu=self.mu,sigma=self.sigma,grid=self.grid.to(x.device))
        # if self.ags==0:
        #     self.ags=torch.tensor(1/(x.numel() * self.ahigh) **0.5)
        
        # self.act_step_size.data=self.act_step_size.clamp(-1,1)
        # print(x.unique(),x.shape,x.div(self.act_step_size).round().unique(),self.act_step_size)
        # 1/0
        return lsq.apply(x,self.act_step_size,torch.tensor(0).to(x.device),self.ahigh.to(x.device),self.ags)
        # return lsq.apply(x,self.act_step_size,self.alow.to(x.device),self.ahigh.to(x.device),self.ags)


class LSQWeightQuantizer(nn.Module):
    
    def __init__(self, w_bits,mean_scale,out_c,binary=False,ternary=True,per_channel=True):
        super(LSQWeightQuantizer, self).__init__()
        self.w_bits=w_bits
        self.tmp_ws=0
        self.w_step_size = nn.Parameter(torch.tensor(1.0))

        if binary:
            self.wlow=torch.tensor(-1.0)
            self.whigh=torch.tensor(1.0)
        elif ternary:
            self.wlow=torch.tensor(-1 **(self.w_bits - 1)) #ternary
            self.whigh=torch.tensor(2 **(self.w_bits-1)-1)
        else:
            self.wlow=torch.tensor(-2 **(self.w_bits - 1))
            self.whigh=torch.tensor(2 **(self.w_bits-1)-1)
        self.mean_scale=mean_scale
        self.ags=torch.tensor(0)
        self.binary=binary
        self.per_channel=per_channel
        self.initialized_alpha=False
    def init_from(self, x, *args, **kwargs):
        # print(x.shape,self.w_step_size.shape)
        if self.per_channel:
            if len(x.shape) == 3 and False:
                init_val = 2 * x.detach().abs().mean(dim=0).mean(dim=0) / (self.whigh ** 0.5)
                print("init:",init_val,x.shape,init_val.shape)
                self.w_step_size.data.copy_(init_val.unsqueeze(1).cuda())
            else:
                init_val = 2 * x.detach().abs().mean() / (self.whigh ** 0.5)
                print("img mush have shape B,C,H,W")
            
                self.w_step_size.data.copy_(init_val.cuda())
        self.initialized_alpha = True
    def forward(self, x):
        
        if (not self.initialized_alpha):
            self.init_from(x)
        return quantize(x.view(-1,x.size(-1)),self.w_step_size,low=self.wlow.to(x.device),high=self.whigh.to(x.device),means=self.mean_scale,weight=True,binary=self.binary)
        # return lsqw.apply(x,self.w_step_size,self.wlow.to(x.device),self.whigh.to(x.device),self.ags)
class LSQKWeightQuantizer(nn.Module):
    def __init__(self, w_bits,in_c,out_c,binary=False,ternary=True):
        super(LSQKWeightQuantizer, self).__init__()
        self.w_bits=w_bits
        self.tmp_ws=0
        self.w_step_size = nn.Parameter(torch.ones(out_c*in_c,1).mul(0.1))
        # if binary:
        #     self.w_step_size = nn.Parameter(torch.tensor(0.1))
        # else: #pretrained
        #     self.w_step_size = nn.Parameter(torch.tensor(0.1))
        if binary:
            self.wlow=torch.tensor(-1.0)
            self.whigh=torch.tensor(1.0)
        elif ternary:
            self.wlow=torch.tensor(-1 **(self.w_bits - 1)) #ternary
            self.whigh=torch.tensor(2 **(self.w_bits-1)-1)
        else:
            self.wlow=torch.tensor(-2 **(self.w_bits - 1))
            self.whigh=torch.tensor(2 **(self.w_bits-1)-1)
        self.mean_scale=False
        self.ags=torch.tensor(0)
        self.binary=binary
    def forward(self, x):
        if False and self.w_step_size.abs()<0.05:
            self.w_step_size.data=self.w_step_size.sign()*torch.tensor(0.05).to(self.w_step_size.device)
        # if self.ags==0:
        #     self.ags=torch.tensor(1/(x.numel() * self.whigh) **0.5)
        if not self.binary:
            self.w_step_size.data=self.w_step_size.clamp(-1,1)
        if False and self.w_step_size==1:
            self.w_step_size.data=self.w_step_size/10
        # print(x.view(x.size(0),-1).norm(dim=1),x.mean(),x.round().unique(),x.div(self.w_step_size).round().unique(),x.shape)
        # print(x.unique(),x.shape,x.div(self.w_step_size).round().unique(),self.w_step_size)
        # 1/0
        return quantize(x,self.w_step_size,low=self.wlow.to(x.device),high=self.whigh.to(x.device),means=self.mean_scale,weight=True,binary=self.binary)
       
class LTQActQuantizer_old(nn.Module):
    def __init__(self, a_bits,mean_scale):
        super(LTQActQuantizer, self).__init__()
        self.a_bits=a_bits
        self.a_step_size = nn.Parameter(torch.tensor(1.0))
        self.whigh=torch.tensor(2 **(self.a_bits - 1))
        self.wlow=-1*torch.tensor(2 **(self.a_bits-1)-1)
        self.mean_scale=mean_scale
        self.ags=torch.tensor(0)
    def forward(self, x):
        # if self.ags==0:
        #     self.ags=torch.tensor(1/(x.numel() * self.whigh) **0.5)
        return lsq.apply(x,self.a_step_size,self.wlow.to(x.device),self.whigh.to(x.device),self.ags)
class LTQActQuantizer(nn.Module):
    def __init__(self, a_bits,mean_scale):
        super(LTQActQuantizer, self).__init__()
        self.a_bits=a_bits
        self.act_step_size =nn.Parameter(torch.tensor(1.0)) #torch.tensor(1.0)# 

        self.ahigh=torch.tensor(2 **(self.a_bits-1)-1)
        self.alow=torch.tensor(-2 **(self.a_bits - 1))
        
        self.mean_scale=mean_scale
        self.ags=torch.tensor(0)
        self.init_batchs=0
      
    def forward(self, x):
        return lsq.apply(x,self.act_step_size,self.alow.to(x.device),self.ahigh.to(x.device),self.ags)
