import torch.backends.cudnn as cudnn
cudnn.benchmark=False

import numpy as np
import lpips
from data import data_loader as dl
import argparse
from IPython import embed
from Test_TestSet import Test_TestSet
from util.visualizer import plot_patches
import random
random.seed(18)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets to train on')
    parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
    parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
    parser.add_argument('--weight_patch', action='store_true', help='compute a weight for each patch')
    parser.add_argument('--fc_on_diff', action='store_true', help='put a few fc layer on top of diff instead of normalizing/averaging')
    parser.add_argument('--weight_output', type=str, default='relu', help='what to do on top of last fc layer for weight patch', choices=['relu','tanh','none'])
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate after FC')
    parser.add_argument('--tanh_score', action='store_true', help='put a tanh on top of FC for scores (force to be in [0,1])')
    parser.add_argument('--weight_multiscale', action='store_true', help='gives all the features to weight branch. If False, gives only last feature map')
    parser.add_argument('--multiview', action='store_true', help='use patches from different views')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')
    parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')
    parser.add_argument('--npatches', type=int, default=65, help='# randomly sampled image patches')
    parser.add_argument('--nInputImg', type=int, default=4, help='# stimuli/images in each batch')
    parser.add_argument('--n_tests', type=int, default=1, help='# stimuli/images in each batch')
    ##missing : new options for weights/fc, nb patches
    parser.add_argument('--model_path', type=str, default=None, help='location of model, will default to ./weights/v[version]/[net_name].pth')
    parser.add_argument('--output_dir', type=str, default=None, help='location of model, will default to ./weights/v[version]/[net_name].pth')

    # parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
    # parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')


    opt = parser.parse_args()
    opt.batch_size = opt.npatches * opt.nInputImg

    # initialize model
    trainer = lpips.Trainer(model=opt.model, net=opt.net, 
        model_path=opt.model_path, use_gpu=opt.use_gpu, gpu_ids=opt.gpu_ids,
        fc_on_diff=opt.fc_on_diff, weight_patch=opt.weight_patch, weight_output=opt.weight_output,
        dropout_rate=opt.dropout_rate, tanh_score=opt.tanh_score, weight_multiscale=opt.weight_multiscale)


    load_size = 64 # default value is 64

    for dataset in opt.datasets:
            data_loader = dl.CreateDataLoader(dataset,dataset_mode='2afc', load_size=load_size, batch_size=opt.batch_size, nThreads=opt.nThreads, Nbpatches= opt.npatches, shuffle=True, multiview=opt.multiview)
            # evaluate model on data
            tester = lpips.Tester(trainer,data_loader)
            # if opt.one_test:
            #     resTestSet  = tester.run_test_set(name=dataset,stop_after=1)
            #     patches, outputs, stimulus = tester.get_current_patches_outputs(opt.nInputImg)
            #     plot_patches(opt.output_dir, 0, patches, outputs, "test_patches", stimulus=stimulus, jitter=not opt.weight_patch)
            # else:
            resTestSet  = tester.run_test_set(name=dataset,stop_after=opt.n_tests, to_plot_patches=True, output_dir=opt.output_dir)
            print('  Dataset [%s]: spearman %.2f'%(dataset,resTestSet['SROCC']))

