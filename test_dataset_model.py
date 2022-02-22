import numpy as np
import lpips
from data import data_loader as dl
import argparse
from IPython import embed
from Test_TestSet import Test_TestSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=['val/traditional','val/cnn','val/superres','val/deblur','val/color','val/frameinterp'], help='datasets to test - for jnd mode: [val/traditional],[val/cnn]; for 2afc mode: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
    parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
    parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')
    parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')
    parser.add_argument('--npatches', type=int, default=65, help='# randomly sampled image patches')
    parser.add_argument('--nInputImg', type=int, default=4, help='# stimuli/images in each batch')
    ##missing : new options for weights/fc, nb patches
    parser.add_argument('--model_path', type=str, default=None, help='location of model, will default to ./weights/v[version]/[net_name].pth')

    # parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
    # parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')


    opt = parser.parse_args()
    opt.batch_size = opt.npatches * opt.nInputImg

    # initialize model
    trainer = lpips.Trainer()
    # trainer.initialize(model=opt.model,net=opt.net,colorspace=opt.colorspace,model_path=opt.model_path,use_gpu=opt.use_gpu)
    trainer.initialize(model=opt.model, net=opt.net, 
        model_path=opt.model_path, use_gpu=opt.use_gpu, gpu_ids=opt.gpu_ids)

    if(opt.model in ['net-lin','net']):
        print('Testing model [%s]-[%s]'%(opt.model,opt.net))
    elif(opt.model in ['l2','ssim']):
        print('Testing model [%s]-[%s]'%(opt.model))


    load_size = 64 # default value is 64

    for dataset in opt.datasets:
            data_loader = dl.CreateDataLoader(dataset,dataset_mode='2afc', load_size=load_size, batch_size=opt.batch_size, nThreads=opt.nThreads, Nbpatches= opt.npatches)

            # evaluate model on data
            resTestSet  = lpips.run_test_set(data_loader, opt, trainer.forward, name=dataset)
            print('  Dataset [%s]: spearman %.2f'%(dataset,resTestSet['SROCC']))

