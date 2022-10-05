import numpy as np
import csv
import glob
import json, os
from PIL import Image

def json_readfile(filename):
    if(os.path.exists(filename)):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        return False
        
def json_savefile(filename, data):
    # store the new json file
    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys=True)

def extract_patches(im_name,out_path,patches_coords):
    ims = [np.asarray(Image.open(im_name%i).convert('RGB')) for i in range(1,5)]
    #print(len(ims),ims[0].shape)
    name = im_name.split("/")[-1][:-4]
    for (p,patch) in enumerate(patches_coords):
        #print(patch)
        imsize=ims[patch[2]-1].shape
        #print(imsize[0])
        im_patch=ims[patch[2]-1][max(patch[1]-patch_size,0):min(patch[1]+patch_size,imsize[0]),max(patch[0]-patch_size,0):min(patch[0]+patch_size,imsize[1])]
        pil_im = Image.fromarray(im_patch)
        pil_im.save(out_path+name+"_P"+str(p+1)+".png") 

patch_size=64+32
patch_size=patch_size//2
data_folder="../../data/dataset/"

patches_coords=json_readfile("patches_coords.json")
for obj in patches_coords:
    print(obj)
    # ref img
    #ref_path = os.path.join(data_folder,"References","VP%d",obj+"_Ref.png")
    #extract_patches(ref_path,os.path.join(data_folder,"References_patches_withVP_threth0.6_bigger/"), patches_coords[obj])
    # distorted ones
    #get list
    distorted=glob.glob(os.path.join(data_folder,"Distorted_Stimuli","VP1",obj+"*.png"))
    for dis in distorted:
        dis=dis.replace("VP1","VP%d")
        extract_patches(dis,os.path.join(data_folder,"PlaylistsStimuli_patches_withVP_threth0.6_bigger/"), patches_coords[obj])