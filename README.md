
# DMWCC: Reproducibility Guide

PyTorch implementation of *Unsupervised Domain Adaptation for Medical Image Segmentation via Dynamic Matrix and Wavelet Consistency Constraints*. DMWCC uses a DeepLabv2 backbone, dynamic matrix feature enhancement (DMFE), wavelet consistency, adversarial alignment, and MK-MMD.

## Environment and installation

The experiments were run on one NVIDIA RTX 4060 Ti GPU. Use a CUDA-enabled PyTorch installation compatible with the local NVIDIA driver.

```bash
conda create -n dmwcc python=3.8 -y
conda activate dmwcc
pip install -r requirements.txt
```

`DeepLab_resnet_pretrained_imagenet.pth` is required by every provided YAML through `TRAIN.RESTORE_FROM`. The optional discriminator/class-center initialization files are referenced by the YAML files under `MPSCL_Pretrained_Model/training/`. Update these absolute paths before running on another machine.

## Dataset preparation

All loaders consume one 2D sample per `.npy` file. Images are normalized to `[-1, 1]`; labels are integer arrays. The loader expands each image to three channels, converts it to BGR, and subtracts the ImageNet mean. Each image-list text file and label-list text file must have the same order and number of lines.

```text
/absolute/path/to/image_000.npy
/absolute/path/to/image_001.npy
```

For MMWHS, place the image and matching `<stem>_gt.npy` label files in the configured locations, then set the four training list paths and four validation-list paths in `scripts/train.py`. `dataset/create_datalist.py` generates paired image/label lists after its four paths are edited. `dataset/create_test_datalist.py` is a legacy helper for `.npz` lists and is not used by `scripts/test.py`.

BraTS and Pro12 list paths are currently declared in `scripts/train.py`; edit the path constants for the local data root. The test-list paths are declared in `scripts/test.py`. Target-domain labels are read by the present data loaders for bookkeeping/evaluation, but they must not be used to optimize the UDA objective.

## Dataset splits and preprocessing

| Dataset | Adaptation tasks | Split used in the manuscript | Preprocessing |
| --- | --- | --- | --- |
| MMWHS17 | CT to MRI; MRI to CT | 20 unpaired CT and 20 unpaired MRI volumes; 16 volumes per modality for training and 4 for testing | SIFA v2 preprocessed data; 2D slices |
| BraTS2018 | T2 to FLAIR; FLAIR to T2 | 1,746 cases per modality; 512 held out for validation/testing | central brain crop to 128 x 128 x 128, then 128 axial slices |
| Pro12 | HK to BIDMC; BIDMC to HK | 12 cases per site; seeded random selection of 10 training and 2 testing cases | axial slices from NIfTI volumes, resized to 256 x 256 |

The MMWHS lists currently in `data/datalist/` contain 2,304 training slices, 576 validation slices, and 4 test-volume entries per modality. Keep splits at the volume level to prevent slices from one subject appearing in both training and testing sets.

## Training

Run commands from `scripts/`. On Windows, select the GPU with `set CUDA_VISIBLE_DEVICES=0`; on Linux/macOS, prepend `CUDA_VISIBLE_DEVICES=0`.

```bash
cd scripts
python train.py --dataset mmwhs --cfg configs/ours_CT2MR.yml
python train.py --dataset mmwhs --cfg configs/ours_MR2CT.yml
python train.py --dataset brats --cfg configs/brats/ours_t22flair.yml
python train.py --dataset brats --cfg configs/brats/ours_flair2t2.yml
python train.py --dataset pro --cfg configs/pro/HK2BIDMC.yml
python train.py --dataset pro --cfg configs/pro/BIDMC2HK.yml
```

The default configuration specifies batch size 4, 50,000 iterations, Adam optimization with learning rate `3e-4`, MK-MMD `kernel_num=5` and `kernel_mul=2.0`, Haar wavelets with 3 levels and enhancement factor 1.5, and adversarial/MK-MMD/consistency weights of 0.003/0.05/0.01, respectively. Run artifacts are saved under `scripts/experiments/snapshots/<source>2<target>/<experiment_name>/`.

## Evaluation

Pass the same YAML used for training so the model architecture and class count match the checkpoint.

```bash
cd scripts
python test.py --cfg configs/ours_CT2MR.yml --dataset mmwhs --target_modality MR --num_class 5 --pretrained_model_pth experiments/snapshots/CT2MR/dmwcc_feature-dmfe_cons-wavelet/model_50000.pth --Method DMWCC
python test.py --cfg configs/brats/ours_t22flair.yml --dataset brats --target_modality flair --num_class 2 --pretrained_model_pth /path/to/brats_checkpoint.pth --Method DMWCC
python test.py --cfg configs/pro/HK2BIDMC.yml --dataset pro --target_modality bidmc --num_class 2 --pretrained_model_pth /path/to/pro_checkpoint.pth --Method DMWCC
```

The evaluator reports mean Dice, Dice standard deviation, mean ASSD, and ASSD standard deviation. It also saves prediction images. Checkpoint filenames depend on `TRAIN.SAVE_PRED_EVERY`; use the checkpoint that exists in the snapshot directory rather than assuming `model_50000.pth`.

## Configuration and random seed

All experiment parameters are YAML files under `scripts/configs/`. `domain_adaptation/config.py` provides defaults; the YAML overrides these defaults. The key experiment fields are `SOURCE`, `TARGET`, `NUM_WORKERS`, `TRAIN.RESTORE_FROM`, `TRAIN.BATCH_SIZE`, `TRAIN.MAX_ITERS`, `TRAIN.feature_enhance_type`, `TRAIN.consistency_aug_type`, and loss weights.

The default random seed is `1234` (`TRAIN.RANDOM_SEED`). Training seeds Python `random`, NumPy, PyTorch CPU, and all CUDA devices; dataloader worker seeds are derived as `1234 + worker_id`. CUDA deterministic mode is not enabled in the current implementation, so minor run-to-run variation may remain. To require deterministic CUDA kernels, set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` before constructing the model, noting the potential speed reduction.

## Notes

This repository contains absolute Windows paths inherited from the experimental environment. They must be changed to valid local paths before training or evaluation. The original training entry point was fixed to BraTS and the test entry point did not load a YAML; both now accept explicit `--dataset` and `--cfg` arguments so the commands above select the correct dataset branch and model configuration.
