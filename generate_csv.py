#now I want to generate a csv file to store a segmentation dataset
import numpy as np
import pandas as pd
import os
# the format is :
#,img,seg
#0, image0_path.jpg, mask0_path.png

#load the image and mask paths
image_dir = "A. Segmentation/1. Original Images/a. Training Set"
image_dir_test = "A. Segmentation/1. Original Images/b. Testing Set"
mask_dir = "A. Segmentation/2. Groundtruths/a. Training Set/1. Intraretinal Microvascular Abnormalities"
mask_dir2 = "A. Segmentation/2. Groundtruths/a. Training Set/2. Nonperfusion Areas"
mask_dir3 = "A. Segmentation/2. Groundtruths/a. Training Set/3. Neovascularization"

#match the name of the image and mask
image_paths = []
mask_paths = []
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    # mask_name = image_name
    # mask_path = os.path.join(mask_dir2, mask_name)
    # if os.path.exists(mask_path):
    #     image_paths.append(image_path)
    #     mask_paths.append(mask_path)
    image_paths.append(image_name)

#generate the csv file
df = pd.DataFrame({"": np.arange(len(image_paths)), "filename": image_paths})
df.to_csv("segmentation_split.csv", index=False)
print("Done!")