import torch.backends.cudnn as cudnn
cudnn.benchmark=False

import numpy as np
import time
import os, sys
import lpips
from data import data_loader as dl
import argparse
from util.visualizer import Visualizer
from IPython import embed
from Test_TestSet import Test_TestSet
import csv
import random
random.seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ##### model/data parameters (same for train or test)
    #dataset
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets to train on')
    parser.add_argument('--multiview', action='store_true', help='use patches from different views')
    #model (do not change)
    parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
    parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
    #for scores
    parser.add_argument('--branch_type', type=str, help='how to get values for each patch: fc or conv', choices=['conv','fc'], default="conv")
    parser.add_argument('--tanh_score', action='store_true', help='put a tanh on top of FC for scores (force to be in [0,1])')
    parser.add_argument('--square_diff', action='store_true', help='square the diff of features (done in LPIPS)')
    parser.add_argument('--normalize_feats', action='store_true', help='normalize the features before doing diff (in LPIPS)')
    parser.add_argument('--nconv', type=int, default=1, help='number of conv in the conv branch')
    #only for weights
    parser.add_argument('--weight_patch', action='store_true', help='compute a weight for each patch')
    parser.add_argument('--weight_output', type=str, default='relu', help='what to do on top of last fc layer for weight patch', choices=['relu','tanh','none'])
    parser.add_argument('--weight_multiscale', action='store_true', help='gives all the features to weight branch. If False, gives only last feature map')
    ##### material stuff
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')
    parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')
    ##### training param
    parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
    parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
    parser.add_argument('--decay_type', type=str, default='divide', help='linear or divide', choices=['divide','linear'])
    parser.add_argument('--npatches', type=int, default=65, help='# randomly sampled image patches')
    parser.add_argument('--nInputImg', type=int, default=4, help='# stimuli/images in each batch')
    parser.add_argument('--data_augmentation', action='store_true', help='use data augmentation on training data')
    parser.add_argument('--lr', type=float, default=0.0001, help='# initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='# beta1 for adam optimizer')
    parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
    parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
    parser.add_argument('--loss', type=str, help='type of loss: L1 or L2', choices=['l1','l2'])
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate after FC')

    parser.add_argument('--use_big_patches', action='store_true', help='use bigger patches (add some randomness)')
    parser.add_argument('--norm_type', type=str, help='normalize patches', choices=['none','mean','unit','lcn'], default="none")
    parser.add_argument('--remove_scaling', action='store_true', help='remove the scaling to adjust to stats of natural images')
    parser.add_argument('--cut_diff2_weights', action='store_true', help='remove squaring of features for weights')
    ##### display/output options
    parser.add_argument('--testset_freq', type=int, default=5, help='frequency of evaluating the testset')
    parser.add_argument('--display_freq', type=int, default=0, help='frequency (in instances) of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=0, help='frequency (in instances) of showing training results on console')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--resume', type=int, default=0, help='checkpoints to resume training')
    parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom display, [0] for no displaying')
    parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
    parser.add_argument('--display_port', type=int, default=8001,  help='visdom display port')
    parser.add_argument('--use_html', action='store_true', help='save off html pages')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory')
    parser.add_argument('--name', type=str, default='tmp', help='directory name for training')
    parser.add_argument('--train_plot', action='store_true', help='plot saving')
    parser.add_argument('--print_net', action='store_true', help='print the network architecture')

    opt = parser.parse_args()
    opt.batch_size = opt.npatches * opt.nInputImg
    
    opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
    if(not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)
    
    original_stdout = sys.stdout
    with open(os.path.join(opt.save_dir,'params.txt'), 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(opt)
        sys.stdout = original_stdout 


    # initialize model
    trainer = lpips.Trainer(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, 
        pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk, gpu_ids=opt.gpu_ids, printNet=opt.print_net,
        is_train=True, lr=opt.lr, beta1=opt.beta1,dropout_rate=opt.dropout_rate, loss=opt.loss,
        norm_type=opt.norm_type, remove_scaling=opt.remove_scaling,
        branch_type=opt.branch_type, tanh_score=opt.tanh_score, normalize_feats=opt.normalize_feats, square_diff=opt.square_diff, nconv=opt.nconv,
        weight_patch=opt.weight_patch, weight_output=opt.weight_output,weight_multiscale=opt.weight_multiscale, cut_diff2_weights=opt.cut_diff2_weights)

    load_size = 64 # default value is 64
    start_epoch=1
    
    train_visualizer = Visualizer(opt, "train")
    test_visualizer = Visualizer(opt, "test")

    if opt.resume > 0:
        trainer.load(opt.save_dir, opt.resume)
        test_visualizer.load_state()
        train_visualizer.load_state()
        start_epoch = opt.resume

    # load data from all test sets 
    # The random patches for the test set are only sampled once at the beginning of training in order to avoid noise in the validation loss.
    dset_name = opt.datasets[0].split("TrainList")[1]
    Testset = os.path.dirname(opt.datasets[0])+'/TexturedDB_20%_TestList'+dset_name                             
    data_loader_testSet = dl.CreateDataLoader(Testset,dataset_mode='2afc', Nbpatches= opt.npatches, data_augmentation=False, use_big_patches=opt.use_big_patches,
                                              load_size = load_size, batch_size=opt.batch_size, nThreads=opt.nThreads, multiview=opt.multiview)
    tester = lpips.Tester(trainer,data_loader_testSet)
    #tester.write_patches()
    #exit()
    

    total_steps = 0
    fid = open(os.path.join(opt.checkpoints_dir,opt.name,'train_log.txt'),'w+')
    f_hyperParam = open(os.path.join(opt.checkpoints_dir,opt.name,'tuning_hyperparam.csv'),'a') 
    if os.stat(os.path.join(opt.checkpoints_dir,opt.name,'tuning_hyperparam.csv')).st_size == 0:
        f_hyperParam.write("nepoch,nepoch_decay,npatches,nInputImg,lr,epoch,TrainLoss,testLoss,SROCC_testset\n")
    
    best_loss = 1e+5
    start_time = time.time()
    for epoch in range(start_epoch, opt.nepoch + opt.nepoch_decay + 1):
        # Load training data to sample random patches every epoch
        data_start_time = time.time()
        data_loader = dl.CreateDataLoader(opt.datasets,dataset_mode='2afc', shuffle=True, Nbpatches=opt.npatches, data_augmentation=opt.data_augmentation, use_big_patches=opt.use_big_patches,
                                            load_size = load_size, batch_size=opt.batch_size, serial_batches=True, nThreads=opt.nThreads, multiview=opt.multiview)
        print(f"Time to load data: {time.time()-data_start_time}")

        dataset = data_loader.load_data()
        dataset_size = len(data_loader) #total number of images, should be equal to nb-batches*batch_size
        nb_batches = len(dataset) #nb of iter to cover all training set

        epoch_start_time = time.time()
        loss_epoch = 0 
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size #total number of patches that went through the net
            epoch_iter = (i+1)*opt.batch_size #total_steps - opt.batch_size*nb_batches * (epoch - 1) # number of images seen in this epoch

            trainer.set_input(data)
            trainer.optimize_parameters()

            errors = trainer.get_current_errors() # current error per batch
            loss_epoch += errors['loss_total'] # total loss over trainset = sum(Loss/batch)/nb_batches



            if opt.display_freq > 0 and total_steps % opt.display_freq == 0:
                train_visualizer.display_current_results(trainer.get_current_visuals(), epoch)

            if opt.print_freq > 0 and total_steps % opt.print_freq == 0:
                t = (time.time()-iter_start_time)/opt.batch_size #time to treat one patch
                t2o = (time.time()-epoch_start_time) #time to do epoch
                t2 = t2o*nb_batches/(i+.0001)
                train_visualizer.print_current_errors(epoch, epoch_iter, errors, t, t2=t2, t2o=t2o, fid=fid)

                for key in errors.keys():
                    train_visualizer.plot_current_errors_save(epoch, float(epoch_iter)/dataset_size, opt, errors, keys=[key,], name=key, to_plot=opt.train_plot)

                if opt.display_id > 0:
                    train_visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)




        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, nb_batches*epoch))
            trainer.save(opt.save_dir, 'latest')
            trainer.save(opt.save_dir, epoch)
            
        loss_epoch = loss_epoch/nb_batches
        print('Epoch Loss %.6f'%loss_epoch)
        resPerEpoch = dict([('loss_total', loss_epoch)])
        for key in resPerEpoch.keys():
            train_visualizer.plot_current_errors_save(epoch, 1.0, opt, resPerEpoch, keys=[key,], name=key, to_plot=opt.train_plot)


        # Evaluate the Test set at the End of the epoch
        info = str(opt.nepoch) + "," + str(opt.nepoch_decay) + "," + str(opt.npatches) + "," + str(opt.nInputImg) + "," + str(opt.lr) + "," + str(epoch) + "," + str(loss_epoch)
        if epoch % opt.testset_freq == 0:
            res_testset = tester.run_test_set(name=Testset) # SROCC & loss
            for key in res_testset.keys():
                test_visualizer.plot_current_errors_save(epoch, 1.0, opt, res_testset, keys=[key,], name=key, to_plot=opt.train_plot)
            if res_testset['loss'] < best_loss:
                best_loss = res_testset['loss']
                trainer.save(opt.save_dir, 'best')
            # patches, outputs = tester.get_current_patches_outputs(2)
            # test_visualizer.plot_patches(epoch, patches, outputs, "patches")
            info += "," + str(res_testset['loss']) + "," + str(res_testset['SROCC'])
        info +=  "\n"
        
        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

        f_hyperParam.write(info)
        
        if (opt.decay_type == 'linear' and epoch > opt.nepoch) or (opt.decay_type == 'divide' and epoch%opt.nepoch_decay == 0):
            trainer.update_learning_rate(opt.nepoch_decay, opt.decay_type)
        

    trainer.save(opt.save_dir, 'latest')
    # trainer.save_done(True)
    fid.close()
    f_hyperParam.close()
    print( 'End of %d epochs. Time taken: %d sec' %(opt.nepoch + opt.nepoch_decay,  time.time() -  start_time))
    
