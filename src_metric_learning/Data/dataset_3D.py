import os
import os.path as osp
from typing import Callable, Optional

from PIL import Image, ImageOps
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import cv2
import warnings
from scipy.ndimage import grey_erosion
from scipy.ndimage.morphology import grey_dilation
from hydra.utils import get_original_cwd


def my_imshow(img, title_str='', cmap='gray'):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    plt.title(title_str)
    plt.show()

class ImgDataset(Dataset):
    def __init__(self,
                 pad_value, norm_value,
                 deviation,
                 curr_seq,
                 type_data,
                 data_dir_img: str,
                 normalize_type: str,
                 data_dir_mask: str,
                 subdir_mask: str,
                 dir_csv: str,
                 num_sequences,
                 use_dilation: bool =True,
                 is_3d: bool = True,
                 train_val_test_split=[80, 10, 10],
                 type_img: str = 'tif',
                 transform: Optional[Callable] = None):
        dir_csv = os.path.join(get_original_cwd(), dir_csv) if dir_csv.startswith('./') else dir_csv
        data_dir_mask = os.path.join(get_original_cwd(), data_dir_mask) if data_dir_mask.startswith('./') else data_dir_mask
        data_dir_img = os.path.join(get_original_cwd(), data_dir_img) if data_dir_img.startswith('./') else data_dir_img
        self.type_data = type_data
        self.split = split = train_val_test_split
        self.num_sequences = num_sequences
        self.deviation = deviation

        self.data_dir_img = data_dir_img
        self.data_dir_mask = data_dir_mask

        self.df_cells = []
        self.images = []
        self.masks = []
        self.frames_all = []
        self.max_cell_id = []
        self.org_df_cells = []
        self.curr_sequence = curr_seq
        # path to csv
        max_cell_id = 0
        for seq in range(self.num_sequences):

            curr_seq_int = curr_seq if self.num_sequences == 1 else seq + 1
            curr_seq_str = "%02d" % curr_seq_int
            dir_csv_curr = osp.join(dir_csv, curr_seq_str)
            dir_csv_curr = osp.join(dir_csv_curr, "csv")
            # read csv
            temp_data = [pd.read_csv(osp.join(dir_csv_curr, file)) for file in sorted(os.listdir(dir_csv_curr))
                         if 'csv' in file]
            curr_df_cells = pd.concat(temp_data, axis=0).reset_index(drop=True)
            self.org_df_cells.append(curr_df_cells)

            if num_sequences > 1 and curr_seq_int > 1:
                curr_df_cells.id = curr_df_cells.id + max_cell_id

            max_cell_id = np.max(curr_df_cells.id)
            self.max_cell_id.append(max_cell_id)
            # path to images and masks
            dir_img = os.path.join(self.data_dir_img, curr_seq_str)
            dir_masks = os.path.join(self.data_dir_mask, f"{curr_seq_str}_{subdir_mask}")
            # read images and masks
            curr_images = [os.path.join(dir_img, fname) for fname in sorted(os.listdir(dir_img)) if type_img in fname]
            curr_masks = [os.path.join(dir_masks, fname) for fname in sorted(os.listdir(dir_masks)) if type_img in fname]

            if len(curr_images) != len(curr_masks):
                min_frames = min(len(curr_images), len(curr_masks))
                curr_images = curr_images[:min_frames]
                curr_masks = curr_masks[:min_frames]
                print(
                    f"Pay attention! the images amd mask are not with the same number {len(curr_images)} != {len(curr_masks)} ")
                for img_path, seg_path in zip(curr_images[:min_frames], curr_masks[:min_frames]):
                    im_num, mask_num = img_path.split(".")[-2][-3:], seg_path.split(".")[-2][-3:]
                    assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"

            assert len(curr_images) == len(curr_masks)

            if self.deviation == 'with_overlap':
                train_val_test_split = np.array(split)
                train_val_test_split = len(curr_images) * train_val_test_split / train_val_test_split.sum()
                train_val_test_split = train_val_test_split.astype('int32')
                train_val_test_split = np.cumsum(train_val_test_split).tolist()
            else:
                un_lables, un_counts = np.unique(curr_df_cells.id, return_counts=True)
                un_counts = 100 * np.cumsum(un_counts) / un_counts.sum()
                np_split = np.array(split).cumsum()
                train_val_test_split = []
                for ind, d_type in enumerate(['train', 'valid', 'test']):
                    curr_precent = np_split[ind]
                    train_val_test_split.append(np.argmin(np.abs(un_counts - curr_precent))+1)

            self.train_val_test_split = train_val_test_split
            curr_splits = [0] + train_val_test_split
            dict_map = {'train': [0, 1], 'valid': [1, 2], 'test': [2, 3]}
            indices = dict_map[type_data]
            if self.deviation == 'with_overlap':
                range_strt, range_stp = curr_splits[indices[0]], curr_splits[indices[1]]
                self.range_list = list(range(range_strt, range_stp))
                mask_df = curr_df_cells.frame_num.isin(self.range_list)
                curr_df_cells = curr_df_cells.loc[mask_df, :]
                self.frames = self.range_list
                self.curr_images = curr_images[self.frames]
                self.curr_masks = curr_masks[self.frames]
            else:
                range_strt, range_stp = curr_splits[indices[0]], curr_splits[indices[1]]
                self.range_list = un_lables[range_strt:range_stp]
                mask_df = curr_df_cells.id.isin(self.range_list)
                curr_df_cells = curr_df_cells.loc[mask_df, :]
                self.frames = np.unique(curr_df_cells.frame_num).tolist()
                self.curr_images = np.array(curr_images)[self.frames].tolist()
                self.curr_masks = np.array(curr_masks)[self.frames].tolist()

            self.frames_all.append(np.array(self.frames))
            self.df_cells.append(curr_df_cells)
            self.images.append(self.curr_images)
            self.masks.append(self.curr_masks)

        self.transform = transform
        self.is_3d = is_3d
        self.normalize_type = normalize_type
        if self.normalize_type == 'regular' and (pad_value is None or norm_value is None):
            self.find_min_max()
        else:
            path_img = curr_images[0]
            img = io.imread(path_img)
            self.img_dtype = img.dtype

        self.pad_value = min(self.min_list) if pad_value is None else pad_value

        if self.normalize_type == 'MinMaxCell':
            assert self.pad_value == 0, f"Problem! The padding value is {self.pad_value} and should be zero !"

        if norm_value is None:
            self.norm_value = max(self.max_list)
        elif norm_value == 'Max':
            if self.img_dtype == 'uint16':
                self.norm_value = 2 ** 16 - 1
            elif self.img_dtype == 'uint8':
                self.norm_value = 2 ** 8 - 1
            else:
                assert False, "Not supported type"
        elif isinstance(norm_value, int):
            self.norm_value = norm_value
        else:
            assert False, "Not supported type"

        self.find_cols()
        self.roi_crop()

        # self.range_list = self.range_list - self.range_list[0]


    def find_min_max(self):
        self.min_list = []
        self.max_list = []
        for curr_sequence in range(self.num_sequences):
            curr_images = self.images[curr_sequence]
            min_list_seq = []
            max_list_seq = []
            for img_path in curr_images:
                img = io.imread(img_path)
                min_list_seq.append(img.min())
                max_list_seq.append(img.max())
            min_np_seq = np.array(min_list_seq)
            max_np_seq = np.array(max_list_seq)
            self.min_list.append(int(min_np_seq.min()))
            self.max_list.append(int(max_np_seq.max()))
        self.img_dtype = img.dtype

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self.__len__() + idx
        cummulative_list_np = np.array(self.cummulative_list)
        curr_sequence = np.argmax(idx < cummulative_list_np)
        frames = self.frames_all[curr_sequence]
        curr_df = self.df_cells[curr_sequence]
        pad_value = self.pad_value  # [curr_sequence]
        norm_value = self.norm_value  # [curr_sequence]
        curr_images = self.images[curr_sequence]
        curr_masks = self.masks[curr_sequence]
        self.max_cell_curr = self.max_cell[curr_sequence]
        self.min_cell_curr = self.min_cell[curr_sequence]
        max_cell_id = self.max_cell_id[curr_sequence - 1] if curr_sequence > 0 else 0

        assert self.max_cell_curr > self.min_cell_curr, "Problem!"

        index = idx
        index -= self.cummulative_list[curr_sequence - 1] if curr_sequence > 0 else 0

        anchor_cell_prop = curr_df.iloc[index, :]

        anchor_cell_frame_ind = int(anchor_cell_prop.frame_num)
        anchor_cell_frame_ind = int(np.where(frames == anchor_cell_frame_ind)[0])

        anchor_cell_id = int(anchor_cell_prop.id)
        anchor_cell_id_real = anchor_cell_id - max_cell_id

        image_curr = io.imread(curr_images[anchor_cell_frame_ind])
        mask_curr = io.imread(curr_masks[anchor_cell_frame_ind])

        bb_anchor = anchor_cell_prop[self.cols].values.squeeze().astype('int32')
        anchor_img_patch = self.crop_norm_padding(image_curr, mask_curr, anchor_cell_id_real, bb_anchor, pad_value)

        min_all = anchor_img_patch.min()
        max_all = anchor_img_patch.max()

        anchor_img_patch = torch.from_numpy(anchor_img_patch).float()

        assert min_all >= 0 and max_all <= 1.0, F"The values [{min_all}, {max_all}] are not in the proper range [0, 1]"

        return anchor_img_patch, anchor_cell_id

    def crop_norm_padding(self, img, mask, id, bb, pad_value):
        if self.is_3d:
            min_row_bb, min_col_bb, max_row_bb, max_col_bb, min_depth_bb, max_depth_bb = bb
            img_patch = img[min_depth_bb:max_depth_bb,
                            min_row_bb:max_row_bb,
                            min_col_bb:max_col_bb].copy()
            msk_patch = mask[min_depth_bb:max_depth_bb,
                             min_row_bb:max_row_bb,
                             min_col_bb:max_col_bb].copy() != id
        else:
            min_row_bb, min_col_bb, max_row_bb, max_col_bb = bb
            img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb].copy()
            msk_patch = mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb].copy() != id

        if not np.any(np.logical_not(msk_patch)):
            warnings.warn("neg sample is all zeros")
            print("neg sample is all zeros")
            print("neg sample is all zeros")
            print("neg sample is all zeros")
        img_patch[msk_patch] = pad_value
        img_patch = img_patch.astype(np.float32)
        if self.normalize_type == 'regular':
            img_patch = self.padding(img_patch, pad_value)[None, ...] / self.norm_value
            img = img_patch
        elif self.normalize_type == 'MinMaxCell':
            not_msk_patch = np.logical_not(msk_patch)
            img_patch[not_msk_patch] = (img_patch[not_msk_patch] - self.min_cell_curr) / (self.max_cell_curr - self.min_cell_curr)
            img = self.padding(img_patch, pad_value)[None, ...]
        else:
            assert False, "Not supported this type of normalization"
        return img

    def padding(self, img, pad_val):
        assert self.is_3d, 'this function is used for 3D'
        desired_size_row = self.curr_roi['row']
        desired_size_col = self.curr_roi['col']
        desired_size_depth = self.curr_roi['depth']

        delta_depth = desired_size_depth - img.shape[0]
        delta_row = desired_size_row - img.shape[1]
        delta_col = desired_size_col - img.shape[2]

        pad_depth = delta_depth // 2
        pad_top = delta_row // 2
        pad_left = delta_col // 2

        image = np.pad(img,
                       ((pad_depth, delta_depth - pad_depth),
                        (pad_top, delta_row - pad_top),
                        (pad_left, delta_col - pad_left)),
                       'constant', constant_values=np.ones((3, 2)) * pad_val)

        return image

    def find_cols(self):
        if self.is_3d:
            self.cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb',
                         'min_depth_bb', 'max_depth_bb']
            self.centers = ["centroid_row", "centroid_col", "centroid_depth"]
        else:
            self.cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb']
            self.centers = ["centroid_row", "centroid_col"]

    def roi_crop(self):
        max_row = 0
        max_col = 0
        max_depth = 0
        self.min_cell = []
        self.max_cell = []
        self.targets = []
        self.frames_for_sampler = []

        for curr_sequence in range(self.num_sequences):

            bb_feat = self.org_df_cells[curr_sequence].loc[:, self.cols]
            max_row = max(max_row, np.abs(bb_feat.min_row_bb.values - bb_feat.max_row_bb.values).max())
            max_col = max(max_col, np.abs(bb_feat.min_col_bb.values - bb_feat.max_col_bb.values).max())
            if self.is_3d:
                max_depth = max(max_depth, np.abs(bb_feat.min_depth_bb.values - bb_feat.max_depth_bb.values).max())

            intensity = self.org_df_cells[curr_sequence].loc[:, ["max_intensity", "min_intensity"]]
            self.min_cell.append(intensity.min_intensity.min())
            self.max_cell.append(intensity.max_intensity.max())

            self.targets.append(self.df_cells[curr_sequence].loc[:, ['id']].values.squeeze())
            self.frames_for_sampler.append(self.df_cells[curr_sequence].loc[:, ['frame_num']].values.squeeze())

        self.targets = np.concatenate(self.targets, axis=0).tolist()
        self.frames_for_sampler = np.concatenate(self.frames_for_sampler, axis=0).tolist()
        self.curr_roi = {'row': max_row, 'col': max_col}
        if self.is_3d:
            self.curr_roi['depth'] = max_depth

    def __len__(self):
        total_num = 0
        self.cummulative_list = []
        for curr_sequence in range(self.num_sequences):
            curr_df = self.df_cells[curr_sequence]
            total_num += curr_df.shape[0]
            self.cummulative_list.append(total_num)
        return total_num
