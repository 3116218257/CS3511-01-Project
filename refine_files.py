#read files in a directory
import os
import sys
import cv2
import SimpleITK
import numpy as np

# def read_files_in_dir(dir_path):
#     files = []
#     for file in os.listdir(dir_path):
#         if file.split("_")[-1] == "post.png":
#             # file_name = file.split("_")[0] + ".png"
#             files.append(file)
#     return files

# new_file_list = read_files_in_dir("submit/task1/class3/train_all_u2net_full/fig")
# print(len(new_file_list))
# print(new_file_list)
# #write new_files in a new directory
# def write_files_in_dir(dir_path, files):
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     for file in files:
#         image = cv2.imread(os.path.join("submit/task1/class3/train_all_u2net_full/fig", file))
#         image_name = file.split("_")[0] + ".png"
#         cv2.imwrite(os.path.join(dir_path, image_name), image)

# write_files_in_dir("class3_submit", new_file_list)

import shutil
import matplotlib.pyplot as plt

def read_nii(nii_path, data_type=np.uint16):
    img = SimpleITK.ReadImage(nii_path)
    data = SimpleITK.GetArrayFromImage(img)
    return np.array(data, dtype=data_type)


def arr2nii(data, filename, reference_name=None):
    img = SimpleITK.GetImageFromArray(data)
    if (reference_name is not None):
        img_ref = SimpleITK.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    SimpleITK.WriteImage(img, filename)


def masks2nii(mask_path,name):
    mask_name_list = os.listdir(mask_path)
    mask_name_list = sorted(mask_name_list, reverse=False, key=lambda x: int(x[:-4]))
    mask_list = []
    for mask_name in mask_name_list:
        mask = cv2.imread(os.path.join(mask_path, mask_name), -1)
        #convert to binary mask with 1 channel
        mask = mask[:, :, 0]
        # save in binary mode
        mask[mask > 0] = 1
        mask_list.append(mask)
    arr2nii(np.array(mask_list), name)
# def read_nii(nii_path, data_type=np.uint16):
#     img = SimpleITK.ReadImage(nii_path)
#     data = SimpleITK.GetArrayFromImage(img)
#     return np.array(data, dtype=data_type)


# def arr2nii(data, filename, reference_name=None):
#     img = SimpleITK.GetImageFromArray(data)
#     if (reference_name is not None):
#         img_ref = SimpleITK.ReadImage(reference_name)
#         img.CopyInformation(img_ref)
#     SimpleITK.WriteImage(img, filename)


# def masks2nii(mask_path):
#     mask_name_list = os.listdir(mask_path)
#     mask_name_list = sorted(mask_name_list, reverse=False, key=lambda x: int(x[:-4]))
#     mask_list = []
#     for mask_name in mask_name_list:
#         mask = cv2.imread(os.path.join(mask_path, mask_name), -1)
#         mask_list.append(mask)
#     arr2nii(np.array(mask_list, np.uint8), "3.nii.gz")


if __name__ == "__main__":
    # path = "class3_submit"
    # masks2nii(path)
    team_name = 'lhy-init'
    if not os.path.exists('./'+team_name):
        os.makedirs('./'+team_name)
    else:
        shutil.rmtree('./'+team_name)
        os.makedirs('./'+team_name)
    path = "class1_submit"
    masks2nii(path,team_name+"/1.nii.gz")
    path = "class2_submit"
    masks2nii(path,team_name+"/2.nii.gz")
    path = "class3_submit"
    masks2nii(path,team_name+"/3.nii.gz")

    if os.path.exists('./'+team_name+'.zip'):
        os.remove('./'+team_name+'.zip')
    os.system('zip -r -q '+team_name+'.zip '+team_name)

    nii = read_nii(team_name+"/1.nii.gz")
    print(nii.shape)