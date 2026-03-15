import numpy as np
import medpy.io as medio


def nii2npz():
    data_nii_pth  ="E:\mmwhs\mr/1009/image.nii"
    label_nii_pth ="E:\mmwhs\mr/1009/label.nii"
    npz_pth       = 'E:\mycode\data\data_np/test_mr\image_mr_1009.nii.gz'

    data_arr, _  = medio.load(data_nii_pth)
    label_arr, _ = medio.load(label_nii_pth)

    np.savez(npz_pth, data_arr, label_arr)


if __name__=="__main__":


    nii2npz()