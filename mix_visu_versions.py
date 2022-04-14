from audioop import mul
import numpy as np
import glob, os
#from skimage import io
from matplotlib import pyplot as plt
from PIL import Image
import tqdm
from mix_R_plots import image_grid, write_on_img_grid
from math import ceil, floor, sqrt

versions = ["checkpoints/YanaParam_40epoch_nodecay_32images_100patches/test/","checkpoints/YanaParam_40epoch_nodecay_32images_100patches/test_best/","checkpoints/YanaParam_40epoch_nodecay_8images_100patches/test_best/","checkpoints/YanaParam_40epoch_nodecay_4images_100patches/test_best/"]
versions = ["checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_32images/test/","checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_32images/test_best/","checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_8images/test_best/","checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images/test_best/"]

versions=["YanaParam_40epoch_nodecay_4images_100patches_dataaug_onlyflips_withWeights","YanaParam_40epoch_nodecay_4images_100patches_dataaug_onlyflips_withWeights_noDiff2","YanaParam_40epoch_nodecay_4images_100patches_dataaug_onlyflips_withWeights_noDiff2Weights","baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images_dataaug_onlyFlips","baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images_onlyflips_l2_beta1_normed"]
versions=["checkpoints/"+v+"/test_best/" for v in versions]
versions_label=["LPIPS both","LPIPS normed","LPIPS normed noDiff2Weights","pieAPP w/o","pieAPP normed"]

nversions=len(versions)
nrow=floor(sqrt(nversions))
ncol=ceil(nversions/nrow)
print(ncol,nrow)

images = sorted(glob.glob(versions[0]+"/test_*.png"))
for im_path in tqdm.tqdm(images):
    im_name=im_path.rsplit('/',1)[1]
    ims = []
    for v in versions:
        if not os.path.exists(v+im_name): 
            print(v+im_name)
            continue
        ims.append (np.asarray(Image.open(v+im_name).convert('RGB')) )
    max_height, max_width = max([im.shape[0] for im in ims]),max([im.shape[1] for im in ims])
    min_height, min_width = min([im.shape[0] for im in ims]),min([im.shape[1] for im in ims])
    #print([im.shape for im in ims], max_height)
    #ims = [np.pad(im,((0,max_height-im.shape[0]),(0,0),(0,0)), constant_values=255)  for im in ims]
    ims = [im[:min_height,-min_width:,:] for im in ims]
    print([im.shape for im in ims])
    #print([im.shape for im in ims])

    final = image_grid(ims,ncol,nrow)
    pil_im = Image.fromarray(final)

    write_on_img_grid(pil_im,ncol,nrow, versions_label)

    pil_im.save(versions[1]+"comp_"+im_name) 
    pil_im.save(versions[0]+"comp_"+im_name)#[:-4]+"2.png")




