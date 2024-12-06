import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Owraper(torch.nn.Module):

    def __init__(self, model,sc):
        super(Owraper, self).__init__()
        self.model = model
        self.max=0
        self.min=0
        self.scale=sc

    def forward(self, data):
        logits = self.model(data)
        tr_loss, tr = l2_reg_ortho(self.model,self.max,self.min,self.scale)  # *1e-2
        if self.model.training:
            return logits, tr_loss, tr
        else:
            return logits
    def load_state_dict(self, state_dict,  strict: bool = True):
        s={}
        hasm=False
        for k,v in state_dict.items():
            if "model." in k:
                hasm=True
            if "module" in k:
                s[k[len("module."):]]=v
            else:
                s[k]=v
        if hasm:
           return  super().load_state_dict(s, strict)
        else:
           return  self.model.load_state_dict(s, strict)
         

def l2_reg_ortho(model,sc):
    rloss = None
    traces = None
    i = 0
    for name, m in model.named_modules():
            # if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) or hasattr(m,"hyper"):
            if hasattr(m,"weight_mat"):
                mma_loss, tr = get_wtwloss(m,sc)
                if rloss is None:
                    rloss = mma_loss
                    traces = tr
                else:
                    rloss = rloss + mma_loss
                    traces = traces + tr
                i += 1
    traces = traces / i
    return rloss, traces


def get_rate(maxepoch, current_epoch, min_r=0.1, max_r=1.0):
    if maxepoch > 0 and current_epoch > 0:
        cut_ratio = min_r + (max_r - min_r) * (1 + np.cos(np.pi * (maxepoch - current_epoch) / maxepoch)) / 2
        return np.float32(cut_ratio)
    else:
        return 0.0
def get_wtwloss(m,scaling):
    if m.weight_mat is None:
        return torch.tensor(0.0, device=m.weight.device), torch.tensor(0.0, device=m.weight.device)  # 0
    weight_mat = m.weight_mat
    # print(weight_mat,m.weight)
    # 1/0
    weight_mat=weight_mat[weight_mat!=0]
    if weight_mat.numel()==0:
        ret=0
    else:
        ret = weight_mat.abs().mean() * 1e-0

    if not weight_mat.sum()==0:
      
        ret=ret*scaling
        return ret, (weight_mat + 1).mean()  # +m.weight.abs().sum()*0.0001
    else:
 
        return ret, 1


def nop_grad(model, masks):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            mk = m._get_name() + str(k)
            if mk in masks:
                mask = masks[mk]
                m.weight.data.mul_(mask)

""" for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d) and args.lockconv:
                _=1
                for param in m.parameters():
                    param.requires_grad = False
            if (isinstance(m,CosDense) or isinstance(m, nn.Conv2d)) and args.lockth:
                #_=1
                #try:
                    if hasattr(m, "clip_val_w_2"):
                        m.clip_val_w_2.requires_grad = False
                        m.clip_val_w_2.mul_(0)
                    if isinstance(m,CosDense):
                        m.clip_val_w_2.cuda()
                        print(m.clip_val_w_2)
#                        1/0
               # except:
               #     _=1
            if isinstance(m, nn.Conv2d) and args.lockconv1:
                _=1
                if m.kernel_size[0]==1:# or m.in_channels==3:
                    for param in m.parameters():
                        param.requires_grad = False
            if isinstance(m, nn.BatchNorm2d) and args.lockbn:
                _=1
                for param in m.parameters():
                    param.requires_grad = False
            if isinstance(m, nn.Linear)  and args.lockfc:
                _=1
                for param in m.parameters():
                    param.requires_grad = False
        for name, param in model.named_parameters():
            print(name,param.requires_grad)#,"cuda:",param.is_cuda)
            if "class" in name and "clip" in name:
                        print(name,param.requires_grad,param)"""

def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def lock_grad(model):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            for param in m.parameters():
                param.requires_grad = False
            if hasattr(m, "clip_pos"):
                m.clip_pos.requires_grad = False
                m.clip_neg.requires_grad = False

    for name, param in model.named_parameters():
            print(name,param.requires_grad)#,"cuda:",param.is_cuda)
            # if "class" in name and "clip" in name:
            #             print(name,param.requires_grad,param)      
def update_trth(model,trth,bylayer,byconv1):
    convth=0
    conv1th=0
    conv3th=0
    if byconv1:
        conv1th = get_thresh(model, nn.Conv2d, trth,idx=1)
        conv3th = get_thresh(model, nn.Conv2d, trth,idx=3)
    else:
        convth = get_thresh(model, nn.Conv2d, trth)
    lineth = get_thresh(model, nn.Linear, trth)
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear): #or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if hasattr(m, 'trth'): 
                if bylayer:
                    m.trth.data=get_thresh_bylayer(m.weight,trth)
                else:
                    m.trth.data = lineth
        if isinstance(m, nn.Conv2d): #or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if hasattr(m, 'trth'): 
                if bylayer:
                    m.trth.data=get_thresh_bylayer(m.weight,trth)
                else:
                    if byconv1:
                        print("DEBUG: trth by conv1(Orig,1,3):",convth,conv1th,conv3th)
                        if m.weight.size()[-1]==1:
                            m.trth.data=conv1th
                        if m.weight.size()[-1]==3:
                            m.trth.data=conv3th
                    else:
                        m.trth.data = convth
def update_scale(model,s):
    for k, m in enumerate(model.modules()):
        if hasattr(m, 'scale_'):
            m.scale_.data = m.scale_.data.mul(0)+s

def turn_on_ste(model,nofc=False):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if hasattr(m, 'ste'): 
                m.ste = True
            if isinstance(m, nn.Linear):
                if hasattr(m, 'ste'):
                    if nofc:
                        m.ste=False
                        m.cos=False
def turn_on_msk(model):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if hasattr(m, 'msk'): 
                m.msk = True
def turn_off_hyper(model):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if hasattr(m, 'hyper'): 
                m.hyper = False
def turn_off_fc(model):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
                if hasattr(m, 'nofc'):
                        m.nofc=True

def turn_on_cos(model):
    for k, m in enumerate(model.modules()):
        #if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if hasattr(m, 'cos'): 
                m.cos = True


def turn_off_ste(model):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if hasattr(m, 'ste'): 
                m.ste = False
def clip(model):
    for k, m in enumerate(model.modules()):
        if hasattr(m, 'clip_w'):
            thre=m.clip_w
            xmask = m.weight.abs().gt(thre.abs()) 
            print(thre,"clipped:",xmask.eq(0).sum()-m.weight.abs().eq(0).sum())
            m.weight.data.mul_(xmask)


def get_thresh(model, instance, percent, idx=-1):
    if percent==0:
        return 0
    total = 0
    for m in model.modules():
        if isinstance(m, instance):
            if m.weight.size()[-1]==idx or idx<0:
                total += m.weight.data.numel()
    if total == 0:
        return 0
    weights = torch.zeros(total)
    index = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, instance):
            if m.weight.size()[-1]==idx or idx<0:
                size = m.weight.data.numel()
                weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                index += size
    y, i = torch.sort(weights)
    thre_index = int(total * percent)
    thre = y[thre_index]
    return thre
def get_thresh_bylayer(weight, percent):
    total = weight.data.numel()
    weights=weight.data.view(-1).abs()
    y, i = torch.sort(weights)
    thre_index = int(total * percent)
    thre = y[thre_index]
    return thre

def recover(model,sp1,sp2,nofc):
    cth2 = get_thresh(model, nn.Conv2d, sp2)
    lth2 = get_thresh(model, nn.Linear, sp2)
    cth1 = get_thresh(model, nn.Conv2d, sp1)
    lth1 = get_thresh(model, nn.Linear, sp1)
    def getmsk(m,th1,th2):
        msk1=m.weight.data.abs().gt(th1).float()
        msk2=m.weight.data.abs().gt(th2).float()
        msk=msk1-msk2
        return msk,msk1,msk2
    def fill(m,zone,msk2):
        if zone.sum()==0:
            
            #print(zone.shape,0,0,zone.sum())
            return 0, zone
        signs=m.weight.sign()
        pruned_=m.weight.data.mul(zone)
        pruned=pruned_[pruned_.abs().gt(0)].flatten()
#        print(m.weight.shape,pruned.numel())
        mea=float(pruned.abs().mean().cpu().detach().numpy())
#        print(signs.shape,mea,pruned.abs().sum(),zone.sum())
        fill=zone*signs*mea
        m.weight.data.mul_(msk2)
#        print("B:",m.weight.data.mean().item(),m.weight.data.std().item())
        m.weight.data.add_(fill)
#        print("A:",m.weight.data.mean().item(),m.weight.data.std().item())
        return mea,fill
#        print(mea,fill,pruned_.sum(),zone.sum())
#        1/0
        
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear) and len(m.weight.shape) == 2:
            if nofc:
                continue
            weight_copy = m.weight.data.abs().clone()
            lrange,msk1,msk2=getmsk(m,lth1,lth2) #get the filling zone
            #_=prune(lth2,m,False)
#            m.weight.data.mul_(msk2)
            mea,ff=fill(m,lrange,msk2)
            #m.weight.data.mul_(msk1)
            #m.weight.data.mul_(1-msk)
            #print("L:",m.weight.shape,mea,lth1,lth2)
         #   print(m.weight.mean().item(),m.weight.std().item(),ff.sum().item())#,lth2,lrange.sum(),mea,m.weight.shape)#,m.weight.view(m.weight.size(0),-1)[0].sum(),weight_copy.view(m.weight.size(0),-1)[0].sum())
        if isinstance(m, nn.Conv2d):
            if m.in_channels == 3:
                continue
            weight_copy = m.weight.data.abs().clone()
            crange,msk1,msk2=getmsk(m,cth1,cth2)
            #_=prune(cth2,m,False)
 #           m.weight.data.mul_(msk2)
            mea,ff=fill(m,crange,msk2)
            #m.weight.data.mul_(msk1)
            #print("C:",m.weight.shape,mea,cth1,cth2)
        #    print(msk1.sum().item(),m.weight.numel(),m.weight.mean().item(),m.weight.std().item(),ff.sum().item())#,cth2,crange.sum(),mea,m.weight.shape)#,m.weight.view(m.weight.size(0),-1)[0].sum(),weight_copy.view(m.weight.size(0),-1)[0].sum())
def prune(thre, m, bins):
    weight_copy = m.weight.data.abs().clone()
    mask = weight_copy.gt(thre).float().cuda()
####    signs=m.weight.sign()
####    pruned_=m.weight.mul(1-mask)
####    pruned=pruned_[pruned_.abs().gt(0)].flatten()
#    print(m.weight.shape,pruned.numel())
####    mea=float(pruned.abs().mean().cpu().detach().numpy())
#    mst=float(pruned.abs().std().cpu().detach().numpy())
#    print(m.weight.shape,mea,thre)
#    filling=torch.empty(pruned_.shape).normal_(mean=0,std=np.abs(mea)).to(m.weight.device)
    #filling=F.normalize(filling.view(m.weight.size(0), -1)).view(m.weight.shape)
    
    #print(filling.mean(),filling.std(),filling.view(filling.size(0),-1).norm(dim=1),m.weight.view(filling.size(0),-1).norm(dim=1))
   # 1/0
    m.weight.data.mul_(mask)
    #rmask=(1-mask)*torch.sqrt(1/m.weight.abs().gt(0).sum())*signs
    #rmask=(1-mask)*m.weight.abs().min()*signs
##best    rmask=(1-mask)*mea*signs
####    rmask=(1-mask)*mea*signs#filling*signs
    #print(rmask[0])
    #1/0
    #print(m.weight.view(filling.size(0),-1))
    #print(m.weight.view(filling.size(0),-1).norm(dim=1))
    #print(rmask.view(filling.size(0),-1))
    #m.weight.data.add_(rmask)
 #   print("B:",m.weight.data.mean().item(),m.weight.data.std().item())
####    m.weight.data.add_(rmask)
 #   print("A:",m.weight.data.mean().item(),m.weight.data.std().item())
 #   print(m.weight.mean().item(),m.weight.std().item(),rmask.sum().item())#,thre,(1-mask).sum(),mea,m.weight.shape)#,m.weight.view(m.weight.size(0),-1)[0].sum(),weight_copy.view(m.weight.size(0),-1)[0].sum())
    #print(m.weight.view(filling.size(0),-1))
    #print(m.weight.view(filling.size(0),-1).norm(dim=1))
    #1/0
    #print(m.weight[0])
    #1/0
#    masks[mk] = mask
    if bins:
        m.weight.data = torch.sign(m.weight)  # to1(m.weight.cpu().detach().numpy()).cuda()
        m.weight.data = F.normalize(m.weight.view(m.weight.size(0), -1)).view(m.weight.shape)
    return mask
def norm(model):
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            if m.in_channels == 3:
                continue
            m.weight.data = F.normalize(m.weight.view(m.weight.size(0), -1)).view(m.weight.shape)
        if isinstance(m, nn.Linear) and len(m.weight.shape) == 2:
            m.weight.data = F.normalize(m.weight.view(m.weight.size(0), -1)).view(m.weight.shape)

def similarity(w,wp):
   w=F.normalize(w.view(w.size(0), -1)) 
   wp=F.normalize(wp.view(wp.size(0), -1))
   s=w@wp.T
   s=s.diag().min()
   #print(w.shape,s.diag(),w.norm(dim=1),wp.norm(dim=1))
   #print(s)
   #1/0
   return s
def c_similarity(w,wp):
   w=w/w.norm()
   wp=wp/wp.norm()
   s=w@wp.T
   #print(w.shape,s.diag(),w.norm(dim=1),wp.norm(dim=1))
   #print(s)
   #1/0
   return s
def dyn_pruning(model,sim,mx):
    _sp=0
    i=0
    if True: # torch.distributed.get_rank()==0:
        for k, m in enumerate(model.modules()):
            if not check(m):
                continue
            dclone=m.weight.clone().data.view(m.weight.size(0),-1)
            for l,wc in enumerate(dclone):
#                print(k,l,wc.shape)
                for sp in range(1,100):
                    lth=get_thresh_bylayer(wc,1-sp/100)
                    w=wc.clone()
                    #print(lth,1-sp/100,w.abs().gt(lth).sum())
        #            print(w.abs().gt(lth))
                    wp=w.mul(w.abs().gt(lth))
                    s=c_similarity(w,wp)
                    if s>=sim:
                        #print("S:",s,lth,1-sp/100,sp)
   #                     m.weight.data=wp
                        dclone[l]=wp
                        #print(wp.abs().gt(0).sum(),dclone.eq(0).sum(),wp,dclone[l])
                        _sp=wp.abs().gt(0).sum()/wp.numel()
                        print(i,l,s)#,_sp,dclone.eq(0).sum())
                        #1/0
                        break
            m.weight.data=dclone.view(m.weight.size())
            if True and False: #torch.distributed.get_rank()==0:
                    print(i,"L:",k,m.weight.shape,"th: %.2f"%lth.item(),"SP:%.2f"%_sp.item(), "%.2f"%s)
            if i>mx:
                break
            i+=1

def check(m):
    if isinstance(m, nn.Conv2d):
        if m.in_channels == 3:
            return False
        else:
            return True
    if isinstance(m, nn.Linear) and len(m.weight.shape) == 2:
        return True

def bin_pruning_v2(model, percent, bins=False,linear_offset=0.0,bylayer=False, nofc=False,byconv1=False,prline=False):
    masks = {}
    convth=0
    conv1th=0
    conv3th=0
    if byconv1:
        conv1th = get_thresh(model, nn.Conv2d, percent,idx=1)
        conv3th = get_thresh(model, nn.Conv2d, percent,idx=3)
    else:
        convth = get_thresh(model, nn.Conv2d, percent)
    if nofc and not prline:
        lineth=0
    else:
        lineth = get_thresh(model, nn.Linear, percent+linear_offset)
    #print(lineth,nofc)
    #1/0
    deconvth = get_thresh(model, nn.ConvTranspose2d, percent)
    for k, m in enumerate(model.modules()):
        mk = m._get_name() + str(k)
        if isinstance(m, nn.Conv2d):
            if m.in_channels == 3:# or m.weight.size(1)==960:
                continue
            if bylayer:
                convth=get_thresh_bylayer(m.weight,percent)
                masks[mk] = prune(convth, m, bins)
                print("Bylayer:",masks[mk].sum())
            else:
                if not byconv1:
                    masks[mk] = prune(convth, m, bins)
                    sp=1-masks[mk].sum()/masks[mk].numel()
                    print("DEBUG: by conv1(Orig,1,3):",convth,conv1th,conv3th,sp.detach().cpu().numpy(),masks[mk].size())
                else:
                    #print("DEBUG: by conv1(Orig,1,3):",convth,conv1th,conv3th)
                    if m.weight.size()[-1]==1:
                        masks[mk] = prune(conv1th, m, bins)
                    if m.weight.size()[-1]==3:
                        masks[mk] = prune(conv3th, m, bins)
                    sp=1-masks[mk].sum()/masks[mk].numel()
                    print("DEBUG: by conv1(Orig,1,3):",convth,conv1th,conv3th,sp.detach().cpu().numpy(),masks[mk].size())
        if isinstance(m, nn.Linear) and len(m.weight.shape) == 2:
            if bylayer:
                lineth=get_thresh_bylayer(m.weight,percent)
            if not nofc or prline:
                masks[mk] = prune(lineth, m, bins)
        if isinstance(m, nn.ConvTranspose2d):
            if bylayer:
                deconvth=get_thresh_bylayer(m.weight,percent)
            masks[mk] = prune(deconvth, m, bins)
    return masks


def summary(model, msg="",sp=False):
    num_parameters_t, ttc, num_parameters_1=0,0,0
    if sp:
        print(msg)
        num_parameters_t = sum([param.nelement() for param in model.parameters()])
        print('Total Parameters: {}'.format(num_parameters_t))
        num_parameters_0, linear_total, am = get_conv_zero_param(model)  # .cpu().detach().numpy()
        ttc = num_parameters_0 + linear_total
        print("\nConv SP:", int(num_parameters_0) / am, am)
        print('Conv total zeros: {}'.format(num_parameters_0))
        num_parameters_10, linear_total, am = get_conv_zero_param(model, [0, 1])  # .cpu().detach().numpy()
        print("\nConv1 SP:", int(num_parameters_10) / am, am)
        print('Conv1 total zeros: {}'.format(num_parameters_10))
        num_parameters_30, linear_total, am = get_conv_zero_param(model, [1, 3])  # .cpu().detach().numpy()
        print("\nConv3 SP:", int(num_parameters_30) / am, am)
        print('Conv3 total zeros: {}'.format(num_parameters_30))
        num_parameters_1 = get_conv_one_param(model).cpu().detach().numpy()
        print('\n\nOne parameters: {}'.format(num_parameters_1))
        print('Sparsity:0-{},1-{},total-{}'.format((ttc) / num_parameters_t, num_parameters_1 / num_parameters_t, (ttc + num_parameters_1) / num_parameters_t))
        num_parameters_k0 = get_conv_zero_filter_param(model).cpu().detach().numpy()
        num_parameters_t0 = get_conv_total_filter_param(model)
        print('Zero-Filter parameters: {}'.format(num_parameters_k0))
        print('Total Zero-Filter parameters: {}'.format(num_parameters_t0))
        print('Filter Sparsity:', num_parameters_k0 / num_parameters_t0)
    c=model.state_dict()
    print("w s:",[c[k].item() for k in c.keys() if "step_size" in k and "act" not in k])
    #print([c[k].item() for k in c.keys() if "clip" in k])
    print("a s:",[c[k].item() for k in c.keys() if "act_step_size" in k])
    return num_parameters_t, ttc, num_parameters_1


def get_conv_zero_param(model, ksize=[0, 3]):
    total = 0
    dtotal = 0
    ltotal = 0
    am = 1
    lm = 1
    dm = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):  # and  (m.kernel_size[0]<=ksize[1] and m.kernel_size[0]>ksize[0]):  # or isinstance(m, nn.Linear):
            total += torch.sum(m.weight.data.eq(0))
            am += m.weight.data.numel()
        if isinstance(m, nn.Linear):
            ltotal += torch.sum(m.weight.data.eq(0))
            lm += m.weight.data.numel()
        if isinstance(m, nn.ConvTranspose2d):
            dtotal += torch.sum(m.weight.data.eq(0))
            dm += m.weight.data.numel()
            
    print("Linear SP:{}".format(int(ltotal) / int(lm)))
    print("Linear total zeros:{}".format(ltotal))
    print("\n\n")
    print("ConvTranspose2d SP:{}".format(int(dtotal) / int(dm)))
    print("ConvTranspose2d total zeros:{}".format(dtotal))
#    print("Conv SP:",ksize, int(total) / am)
    print("w s:",[c[k].item() for k in c.keys() if "step_size" in k and "act" not in k])
    #print([c[k].item() for k in c.keys() if "clip" in k])
    print("a s:",[c[k].item() for k in c.keys() if "act_step_size" in k])
    ass=[]
    for nam,param in model.named_parameters():
        if "act_step_size" in nam:
            ass.append(param.item())
    print("AS:",ass)
    return total.cpu().detach().numpy(), int(ltotal), am


def get_conv_total_filter_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # total += m.weight.view(-1,m.weight.shape[2],m.weight.shape[3]).sum((1,2)).numel()
            total += m.weight.shape[0]  # .view(-1,m.weight.shape[2],m.weight.shape[3]).sum((1,2)).numel()
    return total


def get_conv_zero_filter_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
   #         if m.weight.shape[2]!=1:
                # total += torch.sum(m.weight.view(-1,m.weight.shape[2],m.weight.shape[3]).sum((1,2)).eq(0))
                total += torch.sum(m.weight.reshape(m.weight.shape[0], -1).sum((1)).eq(0))
    return total


def get_conv_one_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            total += torch.sum(m.weight.data.abs().eq(1))
    #        total += torch.sum(m.weight.data.eq(-1))
    return total
