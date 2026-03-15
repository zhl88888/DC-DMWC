"E:\mycode\mmwhs\ct\1001_image.nii"

import nibabel as nib

# 加载 NIfTI 文件
file_path = "E:\mycode\mmwhs\ct/1001_image.nii"  # 替换为你的文件路径
nii_file = nib.load(file_path)

# 获取头信息
header = nii_file.header

# 获取维度信息
dimensions = header.get_data_shape()
print(f"维度: {dimensions}")
