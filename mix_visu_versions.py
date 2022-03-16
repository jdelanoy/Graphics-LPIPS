from audioop import mul
import numpy as np
import glob, os
#from skimage import io
from matplotlib import pyplot as plt
from PIL import Image
import tqdm

versions = ["checkpoints/YanaParam_40epoch_nodecay_32images_100patches/test/","checkpoints/YanaParam_40epoch_nodecay_32images_100patches/test_best/","checkpoints/YanaParam_40epoch_nodecay_8images_100patches/test_best/","checkpoints/YanaParam_40epoch_nodecay_4images_100patches/test_best/"]
versions = ["checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_32images/test/","checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_32images/test_best/","checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_8images/test_best/","checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images/test_best/"]

images = sorted(glob.glob(versions[0]+"/test_*.png"))
for im_path in tqdm.tqdm(images):
    im_name=im_path.rsplit('/',1)[1]
    ims = []
    for v in versions:
        #if not os.path.exists(v+im_name): continue
        ims.append (np.asarray(Image.open(v+im_name).convert('RGB')) )
    max_height = max([im.shape[0] for im in ims])
    #print([im.shape for im in ims], max_height)
    ims = [np.pad(im,((0,max_height-im.shape[0]),(0,0),(0,0)), constant_values=255)  for im in ims]
    #print([im.shape for im in ims])
    image = np.concatenate(ims,axis=1)

    pil_im = Image.fromarray(image)
    pil_im.save(versions[1]+"comp_"+im_name)#[:-4]+"2.png")
