import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import io
import numpy as np
from skimage import io



def read_txt_to_csv(path):
    return pd.read_csv(path, sep=' ', usecols=np.arange(5).tolist(), names=['frame_ind', 'cell_id', 'col', 'row', 'parent_id'])


def like_map_gen(img, frame, cells, save_path):
    debug = True
    black = np.zeros_like(img)
    # likelihood map of one input
    result = black.copy()
    for cell_id, x, y in cells[:, 1:4]:
        img_t = black.copy()  # likelihood map of one cell
        img_t[int(y)][int(x)] = 255  # plot a white dot
        img_t = (gaus_filter(img_t, 51, 6) > 0).astype('int')
        result = np.maximum(result, img_t * cell_id)  # compare result with gaussian_img
    #  normalization
    result = result.astype("uint16")
    save_path = os.path.join(save_path, f"mask_{frame}.tif")
    if debug:
        fig, ax = plt.subplots(2, 2, figsize=(17, 17))
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 1].imshow(result > 0, cmap='gray')
        ax[1, 0].imshow(img, cmap='gray')
        ax[1, 1].imshow(img, cmap='gray')
        ax[1, 1].imshow(result > 0, cmap='gray', alpha=0.5)
        plt.suptitle(frame)
        plt.show()
    else:
        cv2.imwrite(save_path, result)
    print(f"{frame}- n_cells: {cells.shape[0]} - path: {save_path}")


def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(
        img, (pad_size, pad_size), "constant"
    )  # zero padding - Otherwise, after normalization, the likelihood will be brighter near the edge of the image.
    img_t = cv2.GaussianBlur(
        img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma
    )  # filter gaussian(Parameter adjustment as appropriate)
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t


def create_gt_seg(df, path_images, str_seq):
    folder_name = f"exp1_F{str_seq}"
    save_path = os.path.join(path_images, f'../{folder_name}_GT/TRA')
    os.makedirs(save_path, exist_ok=True)
    for ind in np.unique(df.frame_ind):
        ind_str = "%05d" % (ind + 1)
        file_name = f'{folder_name}-{ind_str}.tif'
        path_curr = os.path.join(path_images, file_name)
        img = io.imread(path_curr)
        xx = df.loc[df.frame_ind == ind, :].values
        like_map_gen(img, frame=ind_str, cells=xx, save_path=save_path)


def txt_annotations():
    mapping_dict_training = {
        '0002':
            {
                'gt_txt': "insert/path/to/gt/txt_file",
                'img_path': 'insert/the/corresponding/image/path'
            },
        '0009':
            {
                'gt_txt': "insert/path/to/gt/txt_file",
                'img_path': 'insert/the/corresponding/image/path'
            },
        '0017':
            {
                'gt_txt': "insert/path/to/gt/txt_file",
                'img_path': 'insert/the/corresponding/image/path'
            },
        '0018':
            {
                'gt_txt': "insert/path/to/gt/txt_file",
                'img_path': 'insert/the/corresponding/image/path'
            }
    }

    mapping_dict = mapping_dict_training

    for curr_key in mapping_dict.keys():
        print(f"starts working on: {curr_key}")
        file = mapping_dict[curr_key]['gt_txt']
        path_images = mapping_dict[curr_key]['img_path']

        print(file)
        df = read_txt_to_csv(file)
        print(f"handle the following frame indices: {np.unique(df.frame_ind)}")
        create_gt_seg(df, path_images, str_seq=curr_key)


if __name__ =="__main__":
    txt_annotations()
