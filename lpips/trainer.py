
from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import zoom
from tqdm import tqdm
import lpips
import os
from scipy import stats
import statsmodels.api as sm
import collections
from itertools import groupby
from operator import itemgetter
from statistics import mean


def average_per_stimuli(d0, gt_score, stimulus):
    # In the following: we aggregate gt & d0 per stimulus (over all the patches of the same stimulus)
    predicted_score, patch_weight = d0
    #print(gt_score.flatten())
    #print(predicted_score.flatten())
    #print(patch_weight.flatten())
    judge = (gt_score).flatten().tolist()

    mos = [mean(map(itemgetter(1), group))
        for key, group in groupby(zip(stimulus, judge), key=itemgetter(0))]
    
    NbuniqueStimuli = len(mos) 
    NbpatchesPerStimulus = len(judge)//NbuniqueStimuli # we selected the same nb of patches for each stimulus 
    
    mos = torch.Tensor(mos).to(gt_score.device)
    mos = torch.reshape(mos, (NbuniqueStimuli,1,1,1))
    
    predicted_score_reshaped = torch.reshape(predicted_score, (NbuniqueStimuli,NbpatchesPerStimulus,1,1)) #(5,10,1,1) : 5 stimuli * 10 patches/stimulus => after aggregation : 5 MOS_predicted values
    patch_weight_reshaped = torch.reshape(patch_weight, (NbuniqueStimuli,NbpatchesPerStimulus,1,1))

    product_val = torch.mul(patch_weight_reshaped,predicted_score_reshaped)
    #print("product_val", product_val.shape)
    sum_product_val = torch.sum(product_val, 1, True)
    #print("sum_product_val", sum_product_val.shape)
    norm_factor = torch.sum(patch_weight_reshaped,1,True)
    #print("norm_factor", norm_factor.shape)
    mos_predict = torch.div(sum_product_val, norm_factor)

    #mos_predict = torch.mean(predicted_score_reshaped, 1, True)

    return mos_predict, mos   


class Trainer():
    def name(self):
        return self.model_name

    def initialize(self, model='lpips', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False, weight_patch=False, fc_on_diff=False,
            is_train=False, lr=.001, beta1=0.5, version='0.1', gpu_ids=[0]):
        '''
        INPUTS
            model - ['lpips'] for linearly calibrated network
                    ['baseline'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]'%(model,net)

        if(self.model == 'lpips'): # pretrained net + linear layer
            self.net = lpips.LPIPS(pretrained=not is_train, net=net, version=version, lpips=True, spatial=spatial, 
                pnet_rand=pnet_rand, pnet_tune=pnet_tune, 
                use_dropout=True, model_path=model_path, eval_mode=False, fc_on_diff=fc_on_diff, weight_patch=weight_patch)
        elif(self.model=='baseline'): # pretrained network
            self.net = lpips.LPIPS(pnet_rand=pnet_rand, net=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = lpips.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = lpips.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to map the distance d0 (average over the patches) of the stimulus image to the MOS
            self.loss = lpips.L1Loss()
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        else: # test mode
            self.net.eval()


        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.loss = self.loss.to(device=gpu_ids[0]) # just put this on GPU0

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1(reference)
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''
        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if(hasattr(module, 'weight') and not isinstance(module,nn.Linear) and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_judge = data['judge']
        self.stimulus = data['stimuli_id']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])
            self.stimulus = self.stimulus.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref,requires_grad=True)
        self.var_p0 = Variable(self.input_p0,requires_grad=True)


    def forward_train(self): # run forward pass
        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.var_judge = self.input_judge

        mos_predict, mos = average_per_stimuli(self.d0, self.var_judge, self.stimulus)

        self.loss_total = self.loss.forward(mos_predict, mos) # with aggregation

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward() #torch.mean is useless since we have only one "loss_total" value/batch, and this function is excecuted per batch 

    
    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy())])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256/self.var_ref.data.size()[2]

        ref_img = lpips.tensor2im(self.var_ref.data)
        p0_img = lpips.tensor2im(self.var_p0.data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis)])                   

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self,nepoch_decay,decay_type):
        if (decay_type == 'linear'): 
            lrd = self.lr / nepoch_decay
            lr = self.old_lr - lrd

            for param_group in self.optimizer_net.param_groups:
                param_group['lr'] = lr

            print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
            self.old_lr = lr
        elif (decay_type == 'divide'):
            self.lr = self.lr / 10
            print('update lr [%s] decay:  -> %f' % (type,self.lr))


    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')


def Testset_DSIS(data_loader, opt, func, funcLoss = None, name=''): #added by yana
    total = 0
    SROCC = 0
    val_loss = 0
    val_MSE = 0
    val_steps = 0
    d0s = []
    gts = []
    MOSpredicteds = []
    MOSs = []
    
    
    for data in tqdm(data_loader.load_data(), desc=name):
        with torch.no_grad(): 
            d0 = func(data['ref'],data['p0'])
            gt = data['judge']
            if(opt.use_gpu):
                #d0 = d0.to(device=opt.gpu_ids[0])
                gt = gt.to(device=opt.gpu_ids[0])
            
            stimulus = data['stimuli_id']

            MOSpredicted, MOS = average_per_stimuli(d0, gt, stimulus)
            
            loss = funcLoss(MOSpredicted, MOS) 
            val_loss += loss.cpu().numpy()#detach().numpy() (if we remove "with torch.no_grad():" )
            
            # compute MSE manually
            MSE = ((MOSpredicted-MOS)*(MOSpredicted-MOS)).data.cpu().numpy() 
            val_MSE += np.mean(MSE)
            
            total += gt.size(0)
            val_steps += 1
            
            # concatenate data to compute SROCC
            MOSpredicteds += MOSpredicted
            MOSs += MOS

    #print(MOSpredicteds,MOSs)
    #MOSpredicteds=[mos.cpu().numpy for mos in MOSpredicteds]
    #MOSs=[mos.cpu().numpy for mos in MOSs]
    srocc = 0#stats.spearmanr(MOSpredicteds, MOSs)[0]
    loss = val_loss / val_steps
    MSE = val_MSE / val_steps
    
    print('Testset Total %.3f'%total)
    print('Testset val step = nb batches =  %.3f'%val_steps)
    print('Testset Loss %.3f'%loss)
    print('Testset MSE %.3f'%MSE)
    print('SROCC %.3f'%srocc)
    
    resDict = dict([('loss', loss),
                    ('MSE', MSE),
                    ('SROCC', srocc)])
    return(resDict)

