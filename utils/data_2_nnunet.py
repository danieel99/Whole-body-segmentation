import os
import shutil
import pandas as pd
from pathlib import Path

def make_id(number: int) -> str:
    num_len = len(str(number))
    while num_len < 3:
        number = "0" + str(number)
        num_len = len(number)

    return str(number)

def prepare_data_4_nnunet(data_nifti_path: str, imagesTr_path: str, labels_path: str) -> None:
    i = 0
    csvfile = pd.read_csv('path')
    csvfile2 = pd.read_csv('path')
    for root, folders, files in os.walk(data_nifti_path):
        for file in files:
            file_id = "AutoPET" + make_id(int(i/2))
            print(f'{i}_{os.path.join(root, file)}')
            if "CTres" in file:
                if os.path.join(root,file).replace('file', 'file_name') in csvfile['CT'].values or os.path.join(root,file).replace('file', 'file_name') in csvfile2['CT'].values:
                    file_name = file_id + "_0000.nii.gz"
                    shutil.copy(os.path.join(root, file), os.path.join(imagesTr_path, file_name))
                else:
                    file_name = file_id + "_0000.nii.gz"
                    shutil.copy(os.path.join(root, file), os.path.join(imagesTs_path, file_name))
            if "PET" in file:
                if os.path.join(root,file).replace('file', 'file_name') in csvfile['PET'].values or os.path.join(root,file).replace('file', 'file_name') in csvfile2['PET'].values:
                    file_name = file_id + "_0001.nii.gz"
                    shutil.copy(os.path.join(root, file), os.path.join(imagesTr_path, file_name))
                else:
                    file_name = file_id + "_0001.nii.gz"
                    shutil.copy(os.path.join(root, file), os.path.join(imagesTs_path, file_name))
                
            if "SEG" in file:
                if os.path.join(root,file).replace('file', 'file_name') in csvfile['MASKS'].values or os.path.join(root,file).replace('file', 'file_name') in csvfile2['MASKS'].values:
                    file_name = file_id + ".nii.gz"
                    shutil.copy(os.path.join(root, file), os.path.join(labels_path, file_name))
                else:
                    file_name = file_id + ".nii.gz"
                    shutil.copy(os.path.join(root, file), os.path.join(labelsTs_path, file_name))
            
            # if "SUV" in file:
            #     file_name = file_id + "_0002.nii.gz"
            #     shutil.copy(os.path.join(root, file), os.path.join(imagesTr_path, file_name))
        i += 1

imagesTr_path = "/path/imagesTr"
imagesTs_path = "/path/imagesTs"
labels_path = "/path/labelsTr"
labelsTs_path = "/path/labelsTs"
data_nifti_path = "/path/file"

if __name__ == '__main__':
    prepare_data_4_nnunet(data_nifti_path, imagesTr_path, labels_path)
