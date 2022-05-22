import argparse
import os

#from matplotlib import patches
import lpips
import torch
import numpy as np
import statsmodels.api as sm
from scipy import stats, ndimage
import csv
# from itertools import groupby
# from operator import itemgetter
# from statistics import mean
# from decimal import Decimal
from PIL import Image
import tqdm
from util.visualizer import plot_patches
import torchvision.transforms as transforms
from lpips.trainer import get_img_patches_from_data, get_full_images


def do_all_patches_prediction(net, path, row, multiview, use_gpu, do_plots=False,output_dir=None,weight_patch=False,use_big_patches=False):
    transform_list = []
    transform_list.append(transforms.Resize(64))
    transform_list += [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    dirroots = os.path.dirname(path)+'/'
    root_refPatches = dirroots+'References_patches_withVP_threth0.6'
    root_distPatches = dirroots+'PlaylistsStimuli_patches_withVP_threth0.6'
    if use_big_patches:
        root_refPatches += "_bigger"
        root_distPatches += "_bigger"
    dist = row[1]
    model = row[0]
    MOS = float(row[2])
    nbPatches = int(row[3]) if not multiview else int(row[3]) + int(row[4]) + int(row[5]) + int(row[6])

    im0_patches, im1_patches = [], []
    p0_paths = []

    with torch.no_grad(): 
        for p in range(1, nbPatches +1):
            refpatch = model + '_Ref_P' + str(p) + '.png'
            refpath = os.path.join(root_refPatches, refpatch)
            stimuluspatch = dist + '_P' + str(p) + '.png'
            stimuluspath = os.path.join(root_distPatches, stimuluspatch)
                
            img0 = transform(Image.open(refpath).convert('RGB'))[None]
            img1 = transform(Image.open(stimuluspath).convert('RGB'))[None]

            if(use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            im0_patches.append(img0)
            im1_patches.append(img1)

            p0_paths.append(stimuluspath)

        im0 = torch.cat(im0_patches,0)
        im1 = torch.cat(im1_patches,0)
        score, weight = net.forward(im0,im1)

        MOSpredicted = torch.sum(torch.mul(weight,score), 0, True)/torch.sum(weight,0,True)
                
    if do_plots:
        patches = get_img_patches_from_data(im1, p0_paths, 1, nbPatches)
        images = get_full_images(p0_paths, 1, nbPatches)
        outputs = [score],[weight], [MOSpredicted], [torch.FloatTensor([MOS])]
        plot_patches(output_dir, 0, patches, outputs, f"test_", stimulus=images, have_weight=weight_patch, multiview=multiview)

    return score, weight, MOSpredicted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--csvfile', type=str, default='../../data/dataset/TexturedDB_20%_TestList_withnbPatchesPerVP_threth0.6.csv')
    parser.add_argument('-m','--model_path', type=str, default='./lpips/weights/v0.1/alex.pth', help='location of model')
    parser.add_argument('-o','--output_dir', type=str, default='./GraphicsLPIPS_TestsetScores.csv')
    parser.add_argument('-v','--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    #dataset
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets to train on')
    parser.add_argument('--multiview', action='store_true', help='use patches from different views')
    #model (do not change)
    parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
    parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
    #for scores
    parser.add_argument('--branch_type', type=str, help='how to get values for each patch: fc or conv', choices=['conv','fc'])
    parser.add_argument('--tanh_score', action='store_true', help='put a tanh on top of FC for scores (force to be in [0,1])')
    parser.add_argument('--square_diff', action='store_true', help='square the diff of features (done in LPIPS)')
    parser.add_argument('--normalize_feats', action='store_true', help='normalize the features before doing diff (in LPIPS)')
    parser.add_argument('--nconv', type=int, default=1, help='number of conv in the conv branch')
    #only for weights
    parser.add_argument('--weight_patch', action='store_true', help='compute a weight for each patch')
    parser.add_argument('--weight_output', type=str, default='relu', help='what to do on top of last fc layer for weight patch', choices=['relu','tanh','none'])
    parser.add_argument('--weight_multiscale', action='store_true', help='gives all the features to weight branch. If False, gives only last feature map')

    parser.add_argument('--use_big_patches', action='store_true', help='use bigger patches (add some randomness)')
    parser.add_argument('--norm_type', type=str, help='normalize patches', choices=['none','mean','unit','lcn'], default="none")
    parser.add_argument('--remove_scaling', action='store_true', help='remove the scaling to adjust to stats of natural images')
    parser.add_argument('--cut_diff2_weights', action='store_true', help='remove squaring of features for weights')

    parser.add_argument('--do_plots', action='store_true', help='plot the maps')

    parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')


    opt = parser.parse_args()

    ## Initializing the model
   
        

    loss_fn = lpips.LPIPS(pretrained=True, net=opt.net,
                    use_dropout=True, model_path=opt.model_path, eval_mode=True,dropout_rate=0.0, 
                    branch_type=opt.branch_type, tanh_score=opt.tanh_score, normalize_feats=opt.normalize_feats, square_diff=opt.square_diff, nconv=opt.nconv, norm_type=opt.norm_type, remove_scaling=opt.remove_scaling,
                    weight_patch=opt.weight_patch, weight_output=opt.weight_output,weight_multiscale=opt.weight_multiscale, cut_diff2_weights=opt.cut_diff2_weights)
    if(opt.use_gpu):
        loss_fn.cuda()

    os.makedirs(opt.output_dir,exist_ok=True)

    ## Output file
    f = open(opt.output_dir+"/GraphicsLPIPS_TestsetScores.csv",'w')
    f.writelines('model,p0,simp,qp,qt,size,jpeg,lpips_alex,MOS,std_score,std_weight,mean_score,mean_weight,entropy_score,entropy_weight, spearm corr, pears corr\n')

    ## read Input csv file 
    List_MOS = []
    List_GraphicsLPIPS= []
    List_measures = []

    with open(opt.csvfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in tqdm.tqdm(csv_reader, total=604):
            if line_count == 0:
                line_count += 1
            else:
                dist = row[1]
                model = row[0]
                MOS = float(row[2])
                score,weight, MOSpredicted = do_all_patches_prediction(loss_fn, opt.csvfile, row,opt.multiview, opt.use_gpu, opt.do_plots, opt.output_dir, opt.weight_patch, use_big_patches=opt.use_big_patches)

                #extract all the distorsions
                list_dist = dist.rsplit("_",6)[1:]
                #print(dist,list_dist)
                simp = int(list_dist[0][5:])
                qp = int(list_dist[1][2:])
                qt = int(list_dist[2][2:])
                size = int(list_dist[4].split("x")[1])
                jpeg = int(list_dist[5][1:])

                # 170516_mia337_032122_600_200Kfaces_8192px_OBJ_simpL1_qp9_qt7_decompJPEG_1440x1440_Q10 

                List_GraphicsLPIPS.append(MOSpredicted.item())
                List_MOS.append((MOS))

                res_score_np = [score.item() for score in score]
                res_weight_np = [weight.item() for weight in weight]
                # compute the correlation between scores and weights
                spearm = stats.spearmanr(res_score_np,res_weight_np)[0]
                pears = stats.pearsonr(res_score_np,res_weight_np)[0]
                # compute uniformity: variance and Shannon entropy
                var_score = stats.tstd(res_score_np)
                mean_score = stats.tmean(res_score_np)
                entropy_score = stats.entropy(ndimage.histogram(res_score_np,-1,2,30))
                #print(ndimage.histogram(res_score_np,-1,2,30), entropy_score, var_score)
                var_weight = stats.tstd(res_weight_np)
                mean_weight = stats.tmean(res_weight_np)
                entropy_weight = stats.entropy(ndimage.histogram(res_weight_np,0,max(res_weight_np),50))
                #print(ndimage.histogram(res_weight_np,0,max(res_weight_np),50), entropy_weight, var_weight)
                #List_measures.append({"var_score":var_score,"var_weight":var_weight,"entropy_score":entropy_score,"entropy_weight":entropy_weight})

                f.writelines(f'{model},{dist},{simp},{qp},{qt},{size},{jpeg},{MOSpredicted.item()},{MOS},{var_score},{var_weight},{mean_score},{mean_weight},{entropy_score},{entropy_weight}, {spearm}, {pears}\n')
                line_count +=1
                if line_count > 605: break
    f.close()

    f = open(opt.output_dir+"/GraphicsLPIPS_global.csv",'w')

    List_GraphicsLPIPS = np.array(List_GraphicsLPIPS)
    List_MOS = np.array(List_MOS)

    f.writelines('l1, %.3f\n'%np.mean(np.abs(List_GraphicsLPIPS-List_MOS)))
    f.writelines('l2, %.3f\n'%np.mean((List_GraphicsLPIPS-List_MOS)**2))
    f.writelines('srocc, %.3f\n'%stats.spearmanr(List_GraphicsLPIPS, List_MOS)[0])


    # Instantiate a binomial family model with the logit link function (the default link function).
    List_GraphicsLPIPS = sm.add_constant(List_GraphicsLPIPS)
    glm_binom = sm.GLM(List_MOS, List_GraphicsLPIPS, family = sm.families.Binomial())#, link = sm.families.links.Logit()
    res_regModel = glm_binom.fit()

    fitted_GraphicsLpips = res_regModel.predict()
    corrPears =  stats.pearsonr(fitted_GraphicsLpips, List_MOS)[0]
    corrSpear =  stats.spearmanr(fitted_GraphicsLpips, List_MOS)[0]

    f.writelines('pearson fit, %.3f\n'%corrPears)
    f.writelines('spearman fit, %.3f\n'%corrSpear)




    f.close()


