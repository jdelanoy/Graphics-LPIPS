
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
from itertools import groupby
from operator import itemgetter
from statistics import mean
from PIL import Image
from util.visualizer import plot_patches
import cv2

def get_full_images(patch_paths, nb_images, nb_patches):
    images = []
    for i in range(nb_images):
        path = patch_paths[i*nb_patches]
        dis_path = path.rsplit('_',1)[0].split("/")[-1]
        ref_path = dis_path.rsplit('_simp',1)[0]
        root_folder = path.rsplit('/',2)[0]
        ref_img = np.asarray(Image.open(os.path.join(root_folder,"References/VP1",ref_path+"_Ref.png")).convert('RGB'))
        dis_img1 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP1",dis_path+".png")).convert('RGB'))
        dis_img2 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP2",dis_path+".png")).convert('RGB'))
        dis_img3 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP3",dis_path+".png")).convert('RGB'))
        dis_img4 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP4",dis_path+".png")).convert('RGB'))
        images.append({"path":dis_path, "ref_img": ref_img, "distorted_img1": dis_img1, "distorted_img2": dis_img2, "distorted_img3": dis_img3, "distorted_img4": dis_img4})
    return images

def get_img_patches_from_data(input, patch_paths, nb_images, nb_patches):
    patches = []
    patch_ids = []
    i = 0
    for _ in range(nb_images):
        img_patch = []
        ids = []
        for _ in range(nb_patches):
            if(i >= input.shape[0]) : return patches, patch_ids
            img_patch.append(lpips.tensor2im(input[i:i+1].data))
            ids.append(int(patch_paths[i].rsplit('P',1)[1][:-4]))
            #print(patch_paths[i], ids[-1])
            #plt.imshow(img_patch[-1])
            #plt.show()
            i += 1
        patches.append(img_patch)
        patch_ids.append(ids)
    return patches, patch_ids

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

    mos_predict = torch.sum(torch.mul(patch_weight_reshaped,predicted_score_reshaped), 1, True)/torch.sum(patch_weight_reshaped,1,True)
    #mos_predict = torch.mean(predicted_score_reshaped, 1, True)

    return mos_predict, mos   


class Trainer():
    def name(self):
        return self.model_name

    def __init__(self, model='lpips', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False,
            weight_patch=False, fc_on_diff=False, weight_output='relu', tanh_score = False, dropout_rate=0, weight_multiscale = False,
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
        self.weight_patch = weight_patch

        if(self.model == 'lpips'): # pretrained net + linear layer
            self.net = lpips.LPIPS(pretrained=not is_train, net=net, version=version, lpips=True, spatial=spatial, 
                pnet_rand=pnet_rand, pnet_tune=pnet_tune, 
                use_dropout=True, model_path=model_path, eval_mode=False,
                fc_on_diff=fc_on_diff, weight_patch=weight_patch, weight_output=weight_output, 
                dropout_rate=dropout_rate, tanh_score=tanh_score, weight_multiscale=weight_multiscale)
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
            lpips.print_network(self.net)
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

        ref_img = lpips.tensor2im(self.var_ref[:1].data)
        p0_img = lpips.tensor2im(self.var_p0[:1].data)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis)])                   

    def save(self, path, label):
        if(self.use_gpu):
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
    def load(self, path, label):
        if(self.use_gpu):
            self.load_network(self.net.module, path, '', label)
        else:
            self.load_network(self.net, path, '', label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
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












class Tester():
    def __init__(self, trainer, data_loader):
        self.func = trainer.forward
        self.use_gpu = trainer.use_gpu
        self.gpu_ids = trainer.gpu_ids
        self.weight_patch = trainer.weight_patch
        if trainer.is_train :
            self.funcLoss = trainer.loss.forward
        else:
            self.funcLoss = torch.nn.L1Loss()
        self.data_loader = data_loader

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_judge = data['judge']
        self.stimulus = data['stimuli_id']
        self.path = data['p0_path']

        if(self.use_gpu):
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])
            self.stimulus = self.stimulus.to(device=self.gpu_ids[0])

    def get_current_patches_outputs(self, nb_images, force_update=False):
        if not hasattr(self, 'patches') or force_update:
            #get the patches and the full images only once
            self.patches = get_img_patches_from_data(self.input_p0, self.path, nb_images, self.nb_patches)
            self.images = get_full_images(self.path, nb_images, self.nb_patches)
        return self.patches, self.outputs, self.images

    def write_patches(self):
        for data in self.data_loader.load_data():
            print("writing patches")
            #data = self.data_loader.load_data()[0]
            self.set_input(data)
            patches = get_img_patches_from_data(self.input_p0, self.path, 4, 25)
            for i in range(len(patches[0])):
                for j in range(len(patches[0][i])):
                    #print (patches[0][i][j].shape)
                    cv2.imwrite(f"patches/patch_{i}_{j}.png", patches[0][i][j])
            break

    def run_test_set(self, name='', stop_after = -1, to_plot_patches=False, output_dir=""): #added by yana
        total = 0
        SROCC = 0
        val_loss = 0
        val_MSE = 0
        val_steps = 0

        MOSpredicteds = []
        MOSs = []
        
        for data in tqdm(self.data_loader.load_data(), desc=name):
            with torch.no_grad(): 
                self.set_input(data)
                d0 = self.func(self.input_ref,self.input_p0)
                gt = self.input_judge
                
                stimulus = self.stimulus

                MOSpredicted, MOS = average_per_stimuli(d0, gt, stimulus)
                
                loss = self.funcLoss(MOSpredicted, MOS) 
                val_loss += loss.cpu().numpy()#detach().numpy() (if we remove "with torch.no_grad():" )
                
                # compute MSE manually
                MSE = ((MOSpredicted-MOS)*(MOSpredicted-MOS)).data.cpu().numpy() 
                val_MSE += np.mean(MSE)
                
                total += gt.size(0)
                val_steps += 1
                
                # concatenate data to compute SROCC
                MOSpredicteds += MOSpredicted.flatten().cpu().tolist()
                MOSs += MOS.flatten().cpu().tolist()

                #save the last outputs
                self.nb_patches = int(gt.shape[0]/len(MOS))
                self.outputs = torch.reshape(d0[0], (len(MOS),self.nb_patches)),torch.reshape(d0[1], (len(MOS),self.nb_patches)), MOSpredicted, MOS

                if to_plot_patches:
                    patches, outputs, stimulus = self.get_current_patches_outputs(len(MOS), force_update=True)
                    plot_patches(output_dir, 0, patches, outputs, f"test_", stimulus=stimulus, have_weight=not self.weight_patch)
                    #patches_colormap(output_dir, 0, patches, outputs, f"test_colormap_{val_steps}", stimulus=stimulus, jitter=not self.weight_patch)

                if stop_after > 0 and val_steps>=stop_after: break

        #MOSpredicteds=[mos.flatten().cpu().numpy() for mos in MOSpredicteds]
        #MOSs= MOSs.numpy()[mos.cpu().flatten().numpy() for mos in MOSs]
        srocc = stats.spearmanr(MOSpredicteds, MOSs)[0]
        loss = val_loss / val_steps
        MSE = val_MSE / val_steps
        
        print('Testset number of patches %i'%total)
        print('Testset nb batches =  %i'%val_steps)
        print('Testset Loss %.3f'%loss)
        print('Testset MSE %.3f'%MSE)
        print('SROCC %.3f'%srocc)
        
        measures_dict = dict([('loss', loss),
                        ('MSE', MSE),
                        ('SROCC', srocc)])

        return(measures_dict)

