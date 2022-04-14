from math import ceil, floor, sqrt
import numpy as np
import glob, os
#from skimage import io
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tqdm


def write_on_img_grid(pil_im,ncol,nrow, versions):
    size = pil_im.size[1]/nrow, pil_im.size[0]/ncol
    nversions=len(versions)
    draw = ImageDraw.Draw(pil_im)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf",size=50)
    for i in range(nrow):
        for j in range(ncol):
            if i*ncol+j >= nversions: break
            draw.text((50+j*size[1],50+i*size[0]),f"{versions[i*ncol+j]}",(255,0,0), font=font)
    


def image_grid(image_list,ncol,nrow):
    lines = []
    for i in range(nrow):
        #print(i,image_list[i*ncol:i*ncol+ncol])
        lines.append(np.concatenate(image_list[i*ncol:i*ncol+ncol],axis=1))
    #lines=[np.concatenate(image_list[i*ncol:i*ncol+ncol],axis=1) for i in range(nrow)]
    ## might need to pad last line
    lines[-1] = np.pad(lines[-1],((0,0),(0,lines[0].shape[1]-lines[-1].shape[1]),(0,0)), constant_values=255)
    return np.concatenate(lines,axis=0)



if __name__ == '__main__':

    plots_noweight=["plot_scores_model_mos_lpips_train","plot_scores_model_mos_lpips","plot_unif_score_notweigthed_MOS_std"]
    plots_weighted=["plot_corr_score_weight", "plot_weight_mean","plot_scores_model_mos_weighted_NO","plot_scores_model_mos_weighted",  "plot_scores_model_mos_weighted_train","plot_scores_model_weighted_lpips","plot_unif_score_weigthed_MOS_std","plot_unif_weight_MOS_entropy"]
    plots = plots_weighted


    base="archi_l2_l1_"
    versions = ["without","both","diff2","normed_feat"]
    output="SUMMARY_diff_archi_pieAPP"
    plots = plots_weighted

    base="weighted_lpips_l1_"
    versions = ["without","both","diff2","normed_feat","both_noDiff2Weights"]
    output="SUMMARY_diff_archi_LPIPS"
    plots = plots_weighted+plots_noweight

    # base="weighted_lpips_l1_"
    # versions = ["both","2conv","lastFeat"]
    # output="SUMMARY_diff_morearchi_LPIPS_both"
    # plots = plots_weighted+plots_noweight

    # base="weighted_lpips_l1_"
    # versions = ["normed_feat","2conv_normed","lastFeat_normed"]
    # output="SUMMARY_diff_morearchi_LPIPS_normed"
    # plots = plots_weighted+plots_noweight


    os.makedirs(os.path.join("R analysis",output),exist_ok=True)

    nversions=len(versions)
    nrow=floor(sqrt(nversions))
    ncol=ceil(nversions/nrow)

    for plot_name in plots:
        images=[]
        for v in versions:
            im_path=os.path.join("R analysis",base+v+"_test_best",plot_name+".png")
            if plot_name[-5:] == "train":
                im_path=os.path.join("R analysis",base+v+"_train_best",plot_name[:-6]+".png")
            images.append (np.asarray(Image.open(im_path).convert('RGB')) )
        size=images[-1].shape
        final = image_grid(images,ncol,nrow)
        pil_im = Image.fromarray(final)

        write_on_img_grid(pil_im,ncol,nrow, versions)

        pil_im.save(os.path.join("R analysis",output,plot_name+".png")) 

