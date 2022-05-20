import os.path
import torchvision.transforms as transforms
from data.dataset.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import csv
import random
import collections
# from IPython import embed
import util.transforms2 as T
import cv2

class TwoAFCDataset(BaseDataset):
    def initialize(self, dataroots, load_size=64, shuffle = False, maxNbPatches = 205, multiview=False, data_augmentation=False, use_big_patches=False):
        if(not isinstance(dataroots,list)):
            dataroots = [dataroots,]
        dirroots = os.path.dirname(dataroots[0])+'/'
        root_refPatches = dirroots+'References_patches_withVP_threth0.6'
        root_distPatches = dirroots+'PlaylistsStimuli_patches_withVP_threth0.6'
        if use_big_patches:
            root_refPatches += "_bigger"
            root_distPatches += "_bigger"
        #if(Trainset):           
        #    root_judges = dirroots+'judge_trainingset'
        #else:
        #    root_judges = dirroots+'judge_testset'
        nbiteration = 1
        stimuliId = 0 
        #print(root_refPatches)

        self.load_size = load_size
        
        #shuffle input csv file
        if(shuffle):
            shuffled_inputfile = [] 
            count_inputFile = 0
            print('\t---Shuffling dataset')
            for datafile in dataroots:
                count_inputFile = count_inputFile + 1
                shuffledfileName = dirroots+ 'dataset_shuffled_' + str(count_inputFile) +'.csv'
                shuffled_inputfile.append(shuffledfileName)
                with open(datafile, 'r') as r, open(shuffledfileName, 'w') as w:
                    data = r.readlines()
                    header, rows = data[0], data[1:]
                    random.shuffle(rows)
                    rows = '\n'.join([row.strip() for row in rows])
                    w.write(header + rows)
            
            dataroots = shuffled_inputfile
    
        self.ref_paths = []
        self.p0_paths = []
        self.judge_paths = []
        self.judges = []
        self.stimuliId = []
        
        #print(dataroots)
        for datafile in dataroots:
            with open(datafile) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        #print(f'\tColumn names are {", ".join(row)}')
                        line_count += 1
                    else: # choose patches to load
                        model = row[0]
                        stimulus = row[1]
                        MOS = float(row[2]) 
                        nbPatchesVP1 = int(row[3])
                        nbPatchesVP2 = int(row[4])
                        nbPatchesVP3 = int(row[5])
                        nbPatchesVP4 = int(row[6])
                        nbPatches = nbPatchesVP1 # For "view independent scenario": nbPatches = Total Nb Patches Per Model = nbPatchesVP1 + nbPatchesVP2 + nbPatchesVP3 +nbPatchesVP4
                        if multiview: nbPatches = nbPatchesVP1 + nbPatchesVP2 + nbPatchesVP3 +nbPatchesVP4
                        #judge_npyfile = stimulus + '.npy'
                        #judgepath = os.path.join(root_judges, judge_npyfile)
                       
                        nbfullimage = maxNbPatches//nbPatches
                        nbrandomPatches = maxNbPatches%nbPatches 
                        #print('Nbfull IMAGE %.1f'%nbfullimage)
                        #print('NbRandom patches %.1f'%nbrandomPatches)
                        
                        for itr in range(1, nbiteration+1):
                            stimuliId = stimuliId + 1 
                            for f in range(1, nbfullimage+1):
                                for p in range(1, nbPatches +1):
                                    refpatch = model + '_Ref_P' + str(p) + '.png'
                                    refpath = os.path.join(root_refPatches, refpatch)
                                    self.ref_paths.append(refpath)
                                    stimuluspatch = stimulus + '_P' + str(p) + '.png'
                                    stimuluspath = os.path.join(root_distPatches, stimuluspatch)
                                    self.p0_paths.append(stimuluspath)
                                    #self.judge_paths.append(judgepath) # associate the same judge/MOS to all patches.
                                    self.judges.append(MOS)
                                    self.stimuliId.append(stimuliId) # associate the same StimulusID to all patches.
                            
                            # complete with random patches to reach the max nb of patches
                            for randomPatches in random.sample(range(1,nbPatches+1), nbrandomPatches):
                                refpatch = model + '_Ref_P' + str(randomPatches) + '.png'
                                refpath = os.path.join(root_refPatches, refpatch)
                                self.ref_paths.append(refpath)
                                stimuluspatch = stimulus + '_P' + str(randomPatches) + '.png'
                                stimuluspath = os.path.join(root_distPatches, stimuluspatch)
                                self.p0_paths.append(stimuluspath)
                                #self.judge_paths.append(judgepath)
                                self.judges.append(MOS)
                                self.stimuliId.append(stimuliId)
                        line_count += 1
                        #if line_count == 5: break 
                        
            print(f'\tProcessed {line_count-1} lines (distorted stimuli).')
                
        print('\tTotal nb of patches to load: %.1f' %len(self.p0_paths)) # must be equal to maxNbPatches * nb rows * nb iterations 
        print('\tThese patches correspond to %d stimuli (with repetitions) = %d unique'%(stimuliId, len(set(self.stimuliId))))
        # occurence_stimuliId = collections.Counter(self.stimuliId)
        # print(occurence_stimuliId)

        transform_list = []
        transform_list.append(T.CenterCrop(load_size))
        transform_list += [T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

        self.transform = T.Compose(transform_list)

        if data_augmentation:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(0.5), 
                T.RandomVerticalFlip(0.5),
                T.Random180DegRot(0.5),
                T.Random90DegRotClockWise(0.5),
                #T.Albumentations(50,50,0.5), # change in color
                #T.RandomResize(low=load_size, high=int(load_size*1.2)),
                T.RandomCrop(size=load_size),
                ####T.RandomRotation(degrees=(-5, 5)), 
                self.transform
            ])

        
    
    # # default initialize function of LPIPS
    # def initialize2(self, dataroots, load_size=64):
    #     if(not isinstance(dataroots,list)):
    #         dataroots = [dataroots,]
    #     self.roots = dataroots
    #     self.load_size = load_size
        
    #     # image directory
    #     self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
    #     self.ref_paths = make_dataset(self.dir_ref)
    #     self.ref_paths = sorted(self.ref_paths)

    #     self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
    #     self.p0_paths = make_dataset(self.dir_p0)
    #     self.p0_paths = sorted(self.p0_paths)

    #     transform_list = []
    #     transform_list.append(transforms.Resize(load_size))
    #     transform_list += [transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

    #     self.transform = transforms.Compose(transform_list)

    #     # judgement directory
    #     self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
    #     self.judge_paths = make_dataset(self.dir_J,mode='np')
    #     self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = (cv2.imread(p0_path, 1))#Image.open(p0_path).convert('RGB')
        if p0_img_ is None : print(p0_path)
        p0_img_ = cv2.cvtColor(p0_img_, cv2.COLOR_BGR2RGB)

        ref_path = self.ref_paths[index]
        ref_img_ = cv2.cvtColor(cv2.imread(ref_path, 1), cv2.COLOR_BGR2RGB)# Image.open(ref_path).convert('RGB')
        p0_img, ref_img = self.transform(p0_img_,ref_img_)

        #judge_path = self.judge_paths[index]
        #judge_img = np.load(judge_path).reshape((1,1,1,)) # [0,1]
        #judge_img = torch.FloatTensor(judge_img)

        judge_img = np.array([self.judges[index]]).reshape((1,1,1,)) # [0,1]
        judge_img = torch.FloatTensor(judge_img)
        
        stimuli_id = self.stimuliId[index]
        
        return {'p0': p0_img, 'ref': ref_img, 'judge': judge_img,
            'p0_path': p0_path, 'ref_path': ref_path, 'stimuli_id': stimuli_id}

    def __len__(self):
        return len(self.p0_paths)
