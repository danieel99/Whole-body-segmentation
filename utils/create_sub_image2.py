import json
import os
import torch
import pandas as pd
import torch.utils.tensorboard as tb
from tqdm import tqdm
# from dataloader_tio_aug import AutopetDataloaderTioAug_Zresamplowany, AutopetDataloaderTioAug_Zwykly
import numpy as np
import SimpleITK as sitk


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt
from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN

ROOT_PATH = os.getcwd()

def process_image_dbscan(img, eps=10, min_samples=6):
    coords = np.argwhere(img)
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    bounding_boxes = []
    
    for label in np.unique(labels):
        if label == -1:
            continue
        
        cluster_coords = coords[labels == label]
        
        center = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - center, axis=1)
        max_distance = distances.max()
        if int(max_distance)%32 != 0:
            k = int(max_distance)%32
            max_distance = int(distances.max()) + 32-k

        print(center-max_distance)
        start = np.maximum(center - max_distance, [0, 0, 0]).astype(int)
        end = np.minimum(center + max_distance, img.shape).astype(int)
        
        if (end[0]-start[0])%32 != 0:
            r = (end[0]-start[0])%32
            end[0] += 32-r
        if (end[1]-start[1])%32 != 0:
            r = (end[1]-start[1])%32
            end[1] += 32-r
        if (end[2]-start[2])%32 != 0:
            r = (end[2]-start[2])%32
            end[2] += 32-r

        bounding_box = {
            'center': center,
            'start': start,
            'end': end
        }
        bounding_boxes.append(bounding_box)
    
    return bounding_boxes

def process_image(img):
    labeled_img, num_features = label(img)
    
    bounding_boxes = []
    
    for feature in range(1, num_features+1):
        coords = np.argwhere(labeled_img == feature)
        
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        max_distance = distances.max()
        
        bounding_box = {
            'center': center,
            'side_length': 2 * max_distance
        }
        
        bounding_boxes.append(bounding_box)
    
    return bounding_boxes

def visualize(img, bounding_boxes):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    z,y,x = img.nonzero()
    ax.scatter(x, y, z, zdir='z', c='red', s=1)
    
    for box in bounding_boxes:
        ax.bar3d(box['start'][2], box['start'][1], box['start'][0],
        box['end'][2]-box['start'][2], box['end'][1]-box['start'][1], box['end'][0]-box['start'][0], 
        shade=True, color='cyan', alpha=0.5)
    
    plt.show()

def visualize_2d_slices(img, bounding_boxes, axis=0):
    """
    Visualize 2D slices along the specified axis.
    
    :param img: The 3D image.
    :param bounding_boxes: The list of bounding boxes.
    :param axis: Axis along which to take slices (0 for Z, 1 for Y, 2 for X)
    """
    num_slices = img.shape[axis]
    
    for i in range(num_slices):
        slice_img = np.take(img, indices=i, axis=axis)
        
        plt.figure()
        plt.imshow(slice_img, cmap='gray', origin='lower')
        
        for box in bounding_boxes:
            rect_start = [box['start'][2], box['start'][1]]
            rect_end = [box['end'][2], box['end'][1]]

            if axis == 0:
                rect = plt.Rectangle(rect_start, rect_end[0]-rect_start[0], rect_end[1]-rect_start[1], edgecolor='r', facecolor='none')
            elif axis == 1:
                rect = plt.Rectangle(rect_start, rect_end[0]-rect_start[0], rect_end[1]-rect_start[1], edgecolor='r', facecolor='none')
            else:
                rect = plt.Rectangle(rect_start, rect_end[0]-rect_start[0], rect_end[1]-rect_start[1], edgecolor='r', facecolor='none')

            plt.gca().add_patch(rect)
        
        plt.title(f'Slice {i} along axis {axis}')
        plt.show()

def cut_boxes(img, bounding_boxes):
    sub_images = []
    boxes = []

    for box in bounding_boxes:
        start = box['start']
        end = box['end']
        sub_image = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        sub_images.append(sub_image)
        boxes.append(box)

    return sub_images, boxes

train_data = pd.read_csv("path")
val_data = pd.read_csv("path")
test_data = pd.read_csv("path")

def create_sub_images(data):
  ii = 0
  for i, row in data.iterrows():

    image_ct = sitk.GetArrayFromImage(sitk.ReadImage(row["CT"]))
    image_pet = sitk.GetArrayFromImage(sitk.ReadImage(row["PET"]))
    label = sitk.GetArrayFromImage(sitk.ReadImage(row["MASKS"]))
    output = sitk.GetArrayFromImage(sitk.ReadImage(row["OUTPUT"]))

    # print(len(np.unique(output)))
    output[output >= 0.5] = 1
    output[output < 0.5] = 0

    if len(np.unique(output)) == 2:
        bounding_boxes = process_image_dbscan(output)
        
        if len(bounding_boxes) > 0:
            sub_cts, boxes_cts = cut_boxes(image_ct, bounding_boxes)
            sub_pets, _ = cut_boxes(image_pet, bounding_boxes)
            sub_labels, _ = cut_boxes(label, bounding_boxes)

            save_path = f'save_path'
            if not os.path.exists(os.path.join(save_path, f"patient_{ii+1}")):
                os.makedirs(os.path.join(save_path, f"patient_{ii+1}"))

            orig_ct = f"patient_{ii+1}/orig_ct"
            orig_pet = f"patient_{ii+1}/orig_pet"
            orig_label = f"patient_{ii+1}/orig_label"
            orig_output = f"patient_{ii+1}/orig_output"

            for j in range(len(bounding_boxes)):
                sub_ct = sitk.GetImageFromArray(sub_cts[j])
                sub_pet = sitk.GetImageFromArray(sub_pets[j])
                sub_label = sitk.GetImageFromArray(sub_labels[j])

                ct_path = f"patient_{ii+1}/images/ct_{j+1}"
                pet_path = f"patient_{ii+1}/images/pet_{j+1}"
                label_path = f"patient_{ii+1}/labels/label_{j+1}"

                if not os.path.exists(os.path.join(save_path, f"patient_{ii+1}/images")):
                    os.makedirs(os.path.join(save_path, f"patient_{ii+1}/images"))
                if not os.path.exists(os.path.join(save_path, f"patient_{ii+1}/labels")):
                    os.makedirs(os.path.join(save_path, f"patient_{ii+1}/labels"))

                sitk.WriteImage(sub_ct, os.path.join(save_path, f"{ct_path}.nii.gz"))
                sitk.WriteImage(sub_pet, os.path.join(save_path, f"{pet_path}.nii.gz"))
                sitk.WriteImage(sub_label, os.path.join(save_path, f"{label_path}.nii.gz"))

                for key, value in boxes_cts[j].items():
                    if isinstance(value, np.ndarray):
                        boxes_cts[j][key] = value.tolist()
                        
                with open(os.path.join(save_path, f"{ct_path}.json".replace("ct_", "box_Uniform_")), 'w', encoding='utf-8') as file:
                    json.dump(boxes_cts[j], file, ensure_ascii=False, indent=4)


            sitk.WriteImage(sitk.ReadImage(row["CT"]), os.path.join(save_path, f"{orig_ct}.nii.gz"))
            sitk.WriteImage(sitk.ReadImage(row["PET"]), os.path.join(save_path, f"{orig_pet}.nii.gz"))
            sitk.WriteImage(sitk.GetImageFromArray(output), os.path.join(save_path, f"{orig_output}.nii.gz"))
            sitk.WriteImage(sitk.ReadImage(row["MASKS"]), os.path.join(save_path, f"{orig_label}.nii.gz"))
            ii += 1

if __name__ == '__main__':
    create_sub_images(test_data)
    