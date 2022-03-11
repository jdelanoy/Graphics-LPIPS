import argparse
import os

#from matplotlib import patches
import lpips
import torch
import numpy as np
import statsmodels.api as sm
from scipy import stats
import csv
# from itertools import groupby
# from operator import itemgetter
# from statistics import mean
# from decimal import Decimal
from PIL import Image
import tqdm
from util.visualizer import plot_patches
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--csvfile', type=str, default='../../data/dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv')
parser.add_argument('-m','--model_path', type=str, default='./lpips/weights/v0.1/alex.pth', help='location of model')
parser.add_argument('-o','--output_dir', type=str, default='./GraphicsLPIPS_TestsetScores.csv')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
parser.add_argument('--weight_patch', action='store_true', help='compute a weight for each patch')
parser.add_argument('--fc_on_diff', action='store_true', help='put a few fc layer on top of diff instead of normalizing/averaging')
parser.add_argument('--weight_output', type=str, default='relu', help='what to do on top of last fc layer for weight patch', choices=['relu','tanh','none'])
parser.add_argument('--tanh_score', action='store_true', help='put a tanh on top of FC for scores (force to be in [0,1])')
parser.add_argument('--weight_multiscale', action='store_true', help='gives all the features to weight branch. If False, gives only last feature map')
parser.add_argument('--multiview', action='store_true', help='use patches from different views')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate after FC')

parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')


opt = parser.parse_args()


dirroots = os.path.dirname(opt.csvfile)+'/'
root_refPatches = dirroots+'References_patches_withVP_threth0.6'
root_distPatches = dirroots+'PlaylistsStimuli_patches_withVP_threth0.6'

## Initializing the model

loss_fn = lpips.LPIPS(pretrained=True, net=opt.net,
                use_dropout=True, model_path=opt.model_path, eval_mode=True,
                fc_on_diff=opt.fc_on_diff, weight_patch=opt.weight_patch, weight_output=opt.weight_output, 
                dropout_rate=0.0, tanh_score=opt.tanh_score, weight_multiscale=opt.weight_multiscale)
if(opt.use_gpu):
	loss_fn.cuda()
    
## Output file
f = open(opt.output_dir+"/GraphicsLPIPS_TestsetScores.csv",'w')
f.writelines('p0,lpips_alex,MOS\n')

## read Input csv file 
List_MOS = []
List_GraphicsLPIPS= []

transform_list = []
transform_list.append(transforms.Resize(64))
transform_list += [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

with open(opt.csvfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in tqdm.tqdm(csv_reader, total=604):
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            with torch.no_grad(): 
                dist = row[1]
                model = row[0]
                MOS = float(row[2])
                nbPatches = int(row[3]) if not opt.multiview else int(row[3]) + int(row[4]) + int(row[5]) + int(row[6])
                
                res_score = []
                res_weight = []
                resString =''
                patches_id, patches = [], []
                for p in range(1, nbPatches +1):
                    refpatch = model + '_Ref_P' + str(p) + '.png'
                    refpath = os.path.join(root_refPatches, refpatch)
                    stimuluspatch = dist + '_P' + str(p) + '.png'
                    stimuluspath = os.path.join(root_distPatches, stimuluspatch)
                        

                    img0 = transform(Image.open(refpath).convert('RGB'))
                    img1 = transform(Image.open(stimuluspath).convert('RGB'))
                    img0 = img0[None]
                    img1 = img1[None]
                    #print(img0.shape)
                    #img0 = lpips.im2tensor(lpips.load_image(refpath)) # RGB image from [-1,1]
                    #img1 = lpips.im2tensor(lpips.load_image(stimuluspath))
                    #print(img0.shape)
                    
                    if(opt.use_gpu):
                        img0 = img0.cuda()
                        img1 = img1.cuda()
                    
                    score, weight = loss_fn.forward(img0,img1)
                    res_score.append(score)
                    res_weight.append(weight)

                    #store the patches and everything to launch plot
                    patches.append(lpips.tensor2im(img1.data))
                    patches_id.append(p)
    

            MOSpredicted = sum([score*weight for score,weight in zip(res_score,res_weight)])/sum(res_weight)
            #print(MOSpredicted)
            #MOSpredicted = torch.sum(torch.mul(res_weight,res_score), 1, True)/torch.sum(res_weight,1,True)


            List_GraphicsLPIPS.append(MOSpredicted.item())
            List_MOS.append((MOS))

            outputs = [res_score],[res_weight], [MOSpredicted], [torch.FloatTensor([MOS])]

            dis_path = dist
            ref_path = model
            root_folder = dirroots
            ref_img = np.asarray(Image.open(os.path.join(root_folder,"References/VP1",ref_path+"_Ref.png")).convert('RGB'))
            dis_img1 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP1",dis_path+".png")).convert('RGB'))
            dis_img2 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP2",dis_path+".png")).convert('RGB'))
            dis_img3 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP3",dis_path+".png")).convert('RGB'))
            dis_img4 = np.asarray(Image.open(os.path.join(root_folder,"Distorted_Stimuli/VP4",dis_path+".png")).convert('RGB'))
            images = [{"path":dis_path, "ref_img": ref_img, "distorted_img1": dis_img1, "distorted_img2": dis_img2, "distorted_img3": dis_img3, "distorted_img4": dis_img4}]
            #print((patches, patches_id),outputs)
            plot_patches(opt.output_dir, 0, ([patches], [patches_id]), outputs, f"test_", stimulus=images, have_weight=opt.weight_patch, multiview=opt.multiview)

            f.writelines('%s, %.6f, %s\n'%(dist,MOSpredicted,MOS))
            line_count +=1


List_GraphicsLPIPS = np.array(List_GraphicsLPIPS)
List_MOS = np.array(List_MOS)

f.writelines('l1, %.3f\n'%np.sum(np.abs(List_GraphicsLPIPS-List_MOS)))
f.writelines('l2, %.3f\n'%np.sum((List_GraphicsLPIPS-List_MOS)**2))

# Instantiate a binomial family model with the logit link function (the default link function).
List_GraphicsLPIPS = sm.add_constant(List_GraphicsLPIPS)
glm_binom = sm.GLM(List_MOS, List_GraphicsLPIPS, family = sm.families.Binomial())#, link = sm.families.links.Logit()
res_regModel = glm_binom.fit()

fitted_GraphicsLpips = res_regModel.predict()
corrPears =  stats.pearsonr(fitted_GraphicsLpips, List_MOS)[0]
corrSpear =  stats.spearmanr(fitted_GraphicsLpips, List_MOS)[0]

f.writelines('pearson, %.3f\n'%corrPears)
f.writelines('spearman, %.3f\n'%corrSpear)




f.close()


