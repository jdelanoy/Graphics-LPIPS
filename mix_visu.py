from audioop import mul
import numpy as np
import glob, os
#from skimage import io
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import tqdm
import csv

types=[["baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images","baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images_dataaug","baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images_multiview","baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images_300patches_multiview","baseParam_patch_weight_withrelu_fc_on_diff_dropout_4images_300patches_multiview_dataaug"],["YanaParam_40epoch_nodecay_4images_100patches","YanaParam_40epoch_nodecay_4images_100patches_dataaug","YanaParam_40epoch_nodecay_4images_100patches_multiview","YanaParam_60epoch_nodecay_4images_300patches_multiview","YanaParam_60epoch_nodecay_4images_300patches_multiview_dataaug"]]

#net_with_weights="checkpoints/baseParam_patch_weight_withrelu_fc_on_diff_dropout_32images_multiview/"
#net_no_weights = "checkpoints/YanaParam_40epoch_nodecay_32images_100patches_multiview/"
type_id=1
net_with_weights=os.path.join("checkpoints",types[0][type_id],"test_best/")
net_no_weights = os.path.join("checkpoints",types[1][type_id],"test_best/")
multiview = type_id >= 2
print(net_with_weights, multiview)


#read csv files
csv_weight = csv.reader(open(net_with_weights+"GraphicsLPIPS_TestsetScores.csv"), delimiter=',')
csv_no_weight = csv.reader(open(net_no_weights+"GraphicsLPIPS_TestsetScores.csv"), delimiter=',')

first_line = True
for row1,row2 in tqdm.tqdm(zip(csv_weight,csv_no_weight),total=604):
    if first_line:
        #print(row1)
        first_line=False
        continue
    name = row1[1][0:]
    im_name = "test__0_"+name+".png"
    #print(net_no_weights+im_name)

    if not os.path.exists(net_no_weights+im_name) or  not os.path.exists(net_with_weights+im_name) : continue
    #if os.path.exists(net_with_weights+"test/merge_"+im_name): continue

    im_weight = np.asarray(Image.open(net_with_weights+im_name).convert('RGB')) #io.imread(im_path)
    im_no_weight = np.asarray(Image.open(net_no_weights+im_name).convert('RGB')) #io.imread(net_no_weights+"test/"+im_path.rsplit('/',1)[1])
    #print(im_path)

    if im_no_weight.shape[1] > 1550:
        im_no_weight = im_no_weight[:int(im_no_weight.shape[0]*0.88),-1495:]
    if im_weight.shape[1] > 1550:
        im_weight = im_weight[:int(im_weight.shape[0]*0.88),-1495:] #3093-3522 #2053-2334
    #print(im_no_weight.shape)
    #print(im_weight.shape)

    #multiview = im_weight.shape[0] > 2500

    coords = (490,660) if not multiview else (1740,1400)
    im = np.pad(im_weight,((0,0),(0,coords[1]),(0,0)), constant_values=255)
    im[-coords[0]:,-coords[1]:] = im_no_weight[-coords[0]:,:coords[1]]

    #copy the scores, 25 of height
    #im[40:65,1600:2100] = im_no_weight[40:65,500:1000] #old version
    text_line = 523
    text = im_no_weight[text_line:text_line+23,500:1000]
    text[np.mean(text,axis=2)>100] = 255
    im[text_line:text_line+23,1600:2100] = text
    if multiview:
        im[500:952,-coords[1]:]=im_no_weight[:452,:coords[1]]
        im=im[450:]

    pil_im = Image.fromarray(im)
    draw = ImageDraw.Draw(pil_im)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf",size=32)
    draw.text((1500, 800),f"Weight: entropy {row1[13][:4]}, std {row1[9][:4]}",(0,0,0), font=font)
    draw.text((1500, 850),f"Score  : entropy {row1[12][:4]}, std {row1[8][:4]}",(0,0,0), font=font)
    draw.text((1500, 950),f"Score0: entropy {row2[12][:4]}, std {row2[8][:4]}",(0,0,0), font=font)

    pil_im.save(net_with_weights+"merge_"+im_name)#[:-4]+"2.png")



# for im_path in tqdm.tqdm(images):
#     im_name=im_path.rsplit('/',1)[1]
#     im_weight = np.asarray(Image.open(im_path).convert('RGB')) #io.imread(im_path)
#     im_no_weight = np.asarray(Image.open(net_no_weights+im_name).convert('RGB')) #io.imread(net_no_weights+"test/"+im_path.rsplit('/',1)[1])
#     #print(im_path)

#     if im_no_weight.shape[1] > 1550:
#         im_no_weight = im_no_weight[:int(im_no_weight.shape[0]*0.88),-1495:]
#     if im_weight.shape[1] > 1550:
#         im_weight = im_weight[:int(im_weight.shape[0]*0.88),-1495:] #3093-3522 #2053-2334
#     #print(im_no_weight.shape)
#     #print(im_weight.shape)

#     # # invert the plot and the images
#     # top1, top2 = 915, 1345
#     # images = im_no_weight[top1:top2]
#     # plot = im_no_weight[:top1]
#     # im_no_weight[:top2-top1] = images
#     # im_no_weight[top2-top1:top2] = plot

#     # images = im_weight[top1:top2]
#     # plot = im_weight[:top1]
#     # im_weight[:top2-top1] = images
#     # #im_weight[top2-top1:top2] = plot

#     #multiview = im_weight.shape[0] > 2500

#     coords = (490,660) if not multiview else (1740,1400)
#     im = np.pad(im_weight,((0,0),(0,coords[1]),(0,0)), constant_values=255)

#     im[-coords[0]:,-coords[1]:] = im_no_weight[-coords[0]:,:coords[1]]


#     #copy the scores, 25 of height
#     #im[40:65,1600:2100] = im_no_weight[40:65,500:1000] #old version
#     text_line = 523
#     text = im_no_weight[text_line:text_line+23,500:1000]
#     text[np.mean(text,axis=2)>100] = 255
#     im[text_line:text_line+23,1600:2100] = text
#     if multiview:
#         im[500:952,-coords[1]:]=im_no_weight[:452,:coords[1]]
#         im=im[450:]

#     pil_im = Image.fromarray(im)
#     pil_im.save(net_with_weights+"merge_"+im_name)#[:-4]+"2.png")


