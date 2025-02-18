import torch.utils.data
from data.base_data_loader import BaseDataLoader
import os
import random
import numpy as np
from data.dataset.twoafc_dataset import TwoAFCDataset

# def CreateDataset(dataroots,dataset_mode='2afc',load_size=64, shuffle=False , Nbpatches = 205, multiview=False, data_augmentation=False):
#     dataset = None
#     # Our dataset is baaset on the DSIS protocol (not 2afc). I adapted the code to suit DSIS. However, I did not change the function name.
#     if dataset_mode=='2afc': # human judgements
#         from data.dataset.twoafc_dataset import TwoAFCDataset
#         dataset = TwoAFCDataset()
#     elif dataset_mode=='jnd': # human judgements
#         from data.dataset.jnd_dataset import JNDDataset
#         dataset = JNDDataset()
#     else:
#         raise ValueError("Dataset Mode [%s] not recognized."%dataset_mode)

#     dataset.initialize(dataroots,load_size=load_size,shuffle = shuffle, maxNbPatches = Nbpatches, multiview=multiview, data_augmentation)
#     return dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, data_csvfile, shuffle=False, Nbpatches=205, dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1, multiview=False, data_augmentation=False, use_big_patches=False):
        BaseDataLoader.initialize(self)
        if(not isinstance(data_csvfile,list)):
            data_csvfile = [data_csvfile,]


        g = torch.Generator()
        g.manual_seed(0)

        self.dataset = TwoAFCDataset()
        self.dataset.initialize(data_csvfile,load_size=load_size,shuffle = shuffle, maxNbPatches = Nbpatches, multiview=multiview, data_augmentation=data_augmentation, use_big_patches=use_big_patches)

        #self.dataset = CreateDataset(data_csvfile,dataset_mode=dataset_mode,load_size=load_size, shuffle=shuffle, Nbpatches=Nbpatches, multiview=multiview)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(nThreads),
            worker_init_fn=seed_worker,
            generator=g)
            

    # def initialize2(self, datafolders, dataroot='./dataset',dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1):
    #     BaseDataLoader.initialize(self)
    #     if(not isinstance(datafolders,list)):
    #         datafolders = [datafolders,]
    #     data_root_folders = [os.path.join(dataroot,datafolder) for datafolder in datafolders]
    #     self.dataset = CreateDataset(data_root_folders,dataset_mode=dataset_mode,load_size=load_size)
    #     self.dataloader = torch.utils.data.DataLoader(
    #         self.dataset,
    #         batch_size=batch_size,
    #         shuffle=not serial_batches,
    #         num_workers=int(nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
