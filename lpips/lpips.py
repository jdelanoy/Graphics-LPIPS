
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from . import pretrained_networks as pn
import torch.nn
import torch.nn.functional as F

import lpips

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

def flatten(matrix): # takes NxCxHxW input and outputs NxHWC
    return matrix.view((matrix.shape[0],-1))

def flatten_and_concat(list_tensor): # takes a list of tensors N_i xCxHxW input and outputs NxHWC N = sum N_i
    feats = [flatten(feat) for feat in list_tensor]
    return torch.cat(feats,1) 

# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', # old params (do not use)
            pnet_rand=False, pnet_tune=False, # param about pretrained part of net
            model_path=None, eval_mode=True, verbose=True, # global params
            spatial=False, square_diff=True, normalize_feats=True, branch_type="conv", tanh_score = False, nconv = 1, #score output
            weight_patch=False, weight_output='relu', weight_multiscale = False, # weight output
            use_dropout=True, dropout_rate=0): # training param
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights
        # pnet_rand - random initialization of base network
        # pnet_tune - continue training the base network
        # weight_patch - compute a weight for each patch
        # fc_on_diff - flatten the features and put the difference through FC layers, if False, MSE
        # weight_output - [relu] [tanh] or [none]: the operation applied to the last FC layer for weights
        # dropout_rate - dropout rate (behind FC layers)
        # 

        super(LPIPS, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()
        
        self.weight_patch = weight_patch
        self.branch_type = branch_type
        self.square_diff = square_diff
        self.normalize_feats = normalize_feats
        self.tanh_score = tanh_score
        self.weight_output = weight_output
        self.weight_multiscale = weight_multiscale

        if self.branch_type == "fc":
            self.fc1_score = nn.Linear(31872, 512)
            self.fc2_score = nn.Linear(512,1)
            self.ref_score_subtract = nn.Linear(1,1)
            if self.weight_patch:
                self.fc1_weight = nn.Linear(2304 if not weight_multiscale else 31872,512)
                self.fc2_weight = nn.Linear(512,1)


        self.dropout = nn.Dropout(dropout_rate)

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512] #1472 parameters are learned.
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        #if (branch_type == "conv"):
        self.lins = nn.ModuleList([NetLinLayer(n_channels, use_dropout=use_dropout, nconv=nconv) for n_channels in self.chns])
        if self.weight_patch and branch_type == "conv":
            self.lins_weights = nn.ModuleList([NetLinLayer(n_channels, use_dropout=use_dropout, nconv=nconv) for n_channels in self.chns])
        


        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(pretrained):
            if(model_path is None):
                import inspect
                import os
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))
            if(verbose):
                print('Loading models from: %s'%model_path)
            self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)          

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)

        feats0, feats1, diffs = [], [], []
        if self.normalize_feats:
            for kk in range(self.L):
                feats0.append(lpips.normalize_tensor(outs0[kk]))
                feats1.append(lpips.normalize_tensor(outs1[kk]))
            outs0, outs1=feats0,feats1
        power = 2 if self.square_diff else 1

        #feats0, feats1, diffs = {}, {}, {}

        if self.branch_type == "fc":
            diff = (flatten_and_concat(outs1) - flatten_and_concat(outs0))**power
            val = (self.fc2_score(self.dropout(F.relu(self.fc1_score(diff)))))
            if self.weight_patch:
                if not self.weight_multiscale:
                    diff = (flatten(outs1[-1]) - flatten(outs0[-1]))**power
                per_patch_weight = self.fc2_weight(self.dropout(F.relu(self.fc1_weight(diff))))

        else:
            diffs = [(feats0[kk]-feats1[kk])**power for kk in range(self.L)]

            res = [self.lins[kk](diffs[kk]) for kk in range(self.L)]
            if(self.spatial):
                res = [upsample(res[kk], out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(res[kk], keepdim=True) for kk in range(self.L)]
            val = sum(res)
            # val = res[0]
            # for l in range(1,self.L):
            #     val += res[l]
            if self.weight_patch:
                res = [self.lins_weights[kk](diffs[kk]) for kk in range(self.L)]
                if(self.spatial):
                    res = [upsample(res[kk], out_HW=in0.shape[2:]) for kk in range(self.L)]
                else:
                    res = [spatial_average(res[kk], keepdim=True) for kk in range(self.L)]
                per_patch_weight = sum(res)

        if self.tanh_score:
            val = F.tanh(val)/2+0.5+0.000001

        if self.weight_patch:
            if self.weight_output == 'relu':
                per_patch_weight = F.relu(per_patch_weight)+0.000001
            elif self.weight_output == 'tanh':
                per_patch_weight = F.tanh(per_patch_weight)/2+0.5+0.000001
        else:
            #return weight of 1
            per_patch_weight = torch.ones(val.shape).to(val.device)

#        print(val.shape, per_patch_weight.shape)
        return val, per_patch_weight

        # if(retPerLayer):
        #     return (val, res)
        # else:
        #     return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False, nconv=1):
        super(NetLinLayer, self).__init__()

        #layers = [nn.Dropout(),] if(use_dropout) else []
        #layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        
        chn_mid = chn_in # not used yet
        layers = [nn.Dropout(),] if(use_dropout) else []
        for _ in range(nconv):
            layers += [nn.Conv2d(chn_in, chn_in, 1, stride=1, padding=0, bias=False),]
            layers += [nn.LeakyReLU(0.2,True),]
            layers += [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(1, chn_mid, 1, stride=1, padding=0, bias=True),]#layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,eps=0.1):
        return self.model.forward(d0) 
        #return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1)) # default LPIPS
        

class L1Loss(nn.Module):
    def __init__(self, chn_mid=32):
        super(L1Loss, self).__init__()
        self.loss = torch.nn.L1Loss() #MAE better if we have patches (according to Bosse et al.) 

    def forward(self, d0, judge):
        per = judge
        return self.loss(d0, per)
class L2Loss(nn.Module):
    def __init__(self, chn_mid=32):
        super(L2Loss, self).__init__()
        self.loss = torch.nn.MSELoss() #MAE better if we have patches (according to Bosse et al.) 

    def forward(self, d0, judge):
        per = judge
        return self.loss(d0, per)

# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace

class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = lpips.l2(lpips.tensor2np(lpips.tensor2tensorlab(in0.data,to_norm=False)), 
                lpips.tensor2np(lpips.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = lpips.dssim(1.*lpips.tensor2im(in0.data), 1.*lpips.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = lpips.dssim(lpips.tensor2np(lpips.tensor2tensorlab(in0.data,to_norm=False)), 
                lpips.tensor2np(lpips.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
