import argparse
from glob import glob
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
from matplotlib import pyplot as plt
from lpips_csvFile_TexturedDB import do_all_patches_prediction


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--csvfile', type=str, default='../../data/dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv')
parser.add_argument('-m','--model_path', type=str, nargs='+', help='location of models')
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
parser.add_argument('--do_plots', action='store_true', help='plot the maps')

parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')


opt = parser.parse_args()


losses=np.array([])
l2s = np.array([])
srocc = np.array([])
fits = np.array([])
x=np.array([])

## Initializing the model

for model_path in (opt.model_path):
    print(model_path)
    iter = int(model_path.rsplit("/",1)[1].split("_")[0])
    x = np.append(x,iter)

    loss_fn = lpips.LPIPS(pretrained=True, net=opt.net,
                    use_dropout=True, model_path=model_path, eval_mode=True,
                    fc_on_diff=opt.fc_on_diff, weight_patch=opt.weight_patch, weight_output=opt.weight_output, 
                    dropout_rate=0.0, tanh_score=opt.tanh_score, weight_multiscale=opt.weight_multiscale)
    if(opt.use_gpu):
        loss_fn.cuda()
        

    ## read Input csv file 
    List_MOS = []
    List_GraphicsLPIPS= []


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

                    score, weight, MOSpredicted = do_all_patches_prediction(loss_fn, opt.csvfile,row,opt.multiview, opt.use_gpu)

                    List_GraphicsLPIPS.append(MOSpredicted.item())
                    List_MOS.append((MOS))

                line_count +=1
                #if line_count > 3: break


    List_GraphicsLPIPS = np.array(List_GraphicsLPIPS)
    List_MOS = np.array(List_MOS)


    losses = np.append(losses,np.mean(np.abs(List_GraphicsLPIPS-List_MOS)))
    l2s = np.append(l2s,np.mean((List_GraphicsLPIPS-List_MOS)**2)) 
    srocc = np.append(srocc,stats.spearmanr(List_GraphicsLPIPS, List_MOS)[0])

    # Instantiate a binomial family model with the logit link function (the default link function).
    List_GraphicsLPIPS = sm.add_constant(List_GraphicsLPIPS)
    glm_binom = sm.GLM(List_MOS, List_GraphicsLPIPS, family = sm.families.Binomial())#, link = sm.families.links.Logit()
    res_regModel = glm_binom.fit()
    fitted_GraphicsLpips = res_regModel.predict()
    corrPears =  stats.pearsonr(fitted_GraphicsLpips, List_MOS)[0]
    corrSpear =  stats.spearmanr(fitted_GraphicsLpips, List_MOS)[0]

    fits = np.append(fits,corrPears)

output_path=opt.output_dir

# import os

# #from matplotlib import patches

# from matplotlib import pyplot as plt
# import numpy as np

# types = ["baseParam_patch_weight_withrelu_fc_on_diff_dropout%s","YanaParam_40epoch_nodecay%s_100patches"]
# versions = ["_32images","_8images","_4images"]

# for (it,type) in enumerate(types):
#     for (iv,v) in enumerate(versions):
#         folder = type%v
#         output_path="checkpoints/"+folder+"/test_full"
#         x=np.load(os.path.join(output_path,"loss_x.npy"))
#         losses=np.load(os.path.join(output_path,"loss_y.npy"))
#         l2s=np.load(os.path.join(output_path,"MSE_y.npy"))
#         srocc=np.load(os.path.join(output_path,"SROCC_y.npy"))

sorting=np.argsort(x)
print(x,sorting)
x = x[sorting]
l2s = l2s[sorting]
losses = losses[sorting]
srocc = srocc[sorting]

def save_values(x,y,name,output_path):
    np.save(os.path.join(output_path,name+"_y.npy"),y)
    np.save(os.path.join(output_path,name+"_x.npy"),x)
    plt.plot(x,y)
    plt.savefig(os.path.join(output_path,name+".png"))
    plt.clf()

#output_path=opt.output_dir
os.makedirs(output_path,exist_ok=True)
save_values(x,losses,"loss",output_path)
save_values(x,l2s,"MSE",output_path)
save_values(x,srocc,"SROCC",output_path)
