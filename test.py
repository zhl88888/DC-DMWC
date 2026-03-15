import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__)) 
sys.path.append(curPath)
rootPath = os.path.split(curPath)[0]				
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import argparse
import scipy.io as scio
import warnings

from domain_adaptation.eval_UDA import eval, eval_during_train, load_checkpoint_for_evaluation
from model.deeplabv2 import get_deeplab_v2
from domain_adaptation.config import cfg, cfg_from_file
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")
import numpy as np

def get_arguments():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--pretrained_model_pth', type=str,
        # default='/mnt/workdir/fengwei/ultra_wide/DAUDA_IMAGE/mspcl/scripts/experiments/snapshots/CT2MR1/MT_CT2MR/model_4000.pth',
     default=r"model_1500.pth",
                        help='optional config file', )
    parser.add_argument('--target_modality', type=str, default='MR',
                        help='optional modality', )
    parser.add_argument('--num_class', type=str, default=5, help='number of classes',)
    parser.add_argument('--dataset', type=str, default='mmwhs',
                        help='optional dataset', )
    parser.add_argument('--Method', type=str, default='test',
                        help='optional method', )

    return parser.parse_args()


def main():
    #LOAD ARGS
    args = get_arguments()

    test_list_pth = None
    target_modality = args.target_modality

    if target_modality == 'CT':
        test_list_pth = '\data\datalist/test_ct.txt'

    if target_modality == 'MR':
        test_list_pth = '\data\datalist/test_mr.txt'

    if target_modality == 't2':
        test_list_pth = r'\brats_test_t2.txt'
    if target_modality == 'flair':
        test_list_pth =r'\brats_test_flair.txt'

    if target_modality == 'hk':
        test_list_pth = r'\data\datalist\pro12\hk_test.txt'
    if target_modality == 'bidmc':
        test_list_pth = r'\data\datalist\pro12\bidmc_test.txt'

    with open(test_list_pth) as fp:
        rows = fp.readlines()
    testfile_list = [row[:-1] for row in rows]

    model = None
    if cfg.TRAIN.MODEL == 'DeepLabv2':

        model = get_deeplab_v2(num_classes = args.num_class,multi_level=cfg.TRAIN.MULTI_LEVEL)


    pretrained_model_pth  = args.pretrained_model_pth
    load_checkpoint_for_evaluation(model, pretrained_model_pth)
    Method = args.Method
    print('target_modality is {},method is {}'.format(target_modality,args.Method))
    #dice_mean,dice_std,assd_mean,assd_std,ece_value = eval(model,testfile_list,target_modality,pretrained_model_pth,Method, save_img=True)
    dice_mean, dice_std, assd_mean, assd_std= eval(model, testfile_list, target_modality,
                                                               pretrained_model_pth, Method, save_img=True,dataset=args.dataset)


    print(dice_mean.mean(), dice_std.mean(), assd_mean.mean(), assd_std.mean())

if __name__ == '__main__':
    main()
