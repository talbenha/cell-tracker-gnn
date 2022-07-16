import os
import os.path as op
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import warnings
from hydra.utils import get_original_cwd, to_absolute_path

from src_metric_learning.modules.resnet_2d.resnet import set_model_architecture, MLP

#
class TestDataset(Dataset):

    def __init__(self,
                 debug,
                 path: str,
                 sec_path: str,
                 path_marker: str,
                 type_img: str,):
        path = os.path.join(get_original_cwd(), path) if path.startswith('./') else path
        path_marker = os.path.join(get_original_cwd(), path_marker) if path_marker.startswith('./') else path_marker

        self.debug = debug
        self.path = path
        self.sec_path = sec_path

        type_masks = type_img
        dir_img = path
        dir_masks = path_marker

        assert os.path.exists(dir_img), "Image paths is not exist, please fix it!"
        assert os.path.exists(dir_masks), "Masks paths is not exist, please fix it!"

        self.images = []
        if os.path.exists(dir_img):
            self.images = [os.path.join(dir_img, fname) for fname in sorted(os.listdir(dir_img))
                           if type_img in fname]
        self.masks = []
        if os.path.exists(dir_masks):
            self.masks = [os.path.join(dir_masks, fname) for fname in sorted(os.listdir(dir_masks))
                          if type_masks in fname]

    def __getitem__(self, idx):
        assert len(self.images) or len(self.images), "both directories are empty, please fix it!"

        im_path, image = None, None
        if len(self.images):
            im_path = self.images[idx]
            image = np.array(Image.open(im_path))

        mask_path, mask = None, None
        if len(self.masks):
            mask_path = self.masks[idx]
            mask = np.array(Image.open(mask_path))

        flag = True
        if im_path is not None:
            flag = False
            im_num = im_path.split(".")[-2][-3:]
        if mask_path is not None:
            flag = False
            mask_num = mask_path.split(".")[-2][-3:]

        if flag:
            assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"

        return image, mask, im_path, mask_path

    def __len__(self):
        return len(self.images)

    def extract_patch(self, img, bbox, col, row, cell_id=None):
        kernel = (64, 64)
        min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
        d_row = kernel[0] - (max_row_bb - min_row_bb)
        d_col = kernel[1] - (max_col_bb - min_col_bb)
        shape_row, shape_col = img.shape

        # find cols min max
        if min_row_bb < (d_row // 2):
            min_row = 0
            max_row = kernel[0]
        elif (max_row_bb + d_row // 2) > shape_row:
            min_row = shape_row - kernel[0] - 1
            max_row = shape_row - 1
        else:
            min_row = min_row_bb - d_row // 2
            max_row = max_row_bb + d_row // 2
            if (d_row // 2) * 2 != d_row:
                if min_row == 0:
                    max_row += 1
                else:
                    min_row -= 1

        # find cols min max
        if min_col_bb < (d_col // 2):
            min_col = 0
            max_col = kernel[1]
        elif (max_col_bb + d_col // 2) > shape_col:
            min_col = shape_col - kernel[1] - 1
            max_col = shape_col - 1
        else:
            min_col = min_col_bb - d_col // 2
            max_col = max_col_bb + d_col // 2
            if (d_col // 2) * 2 != d_col:
                if min_col == 0:
                    max_col += 1
                else:
                    min_col -= 1


        patch = img[min_row: max_row, min_col: max_col]
        assert patch.shape == kernel, f"patch.shape: {patch.shape} , " \
            f"[min_row, max_row, min_col,  max_col] = {[min_row, max_row, min_col, max_col]}"
        if self.debug:
            fig, ax = plt.subplots(1, 2, figsize=(17, 6))
            ax[0].imshow(patch, cmap='gray')
            ax[1].imshow(img, cmap='gray')
            ax[1].plot(col, row, "rx")
            plt.show()
        return min_row, max_row, min_col, max_col

    def padding(self, img):
        desired_size_row = self.roi_model['row']
        desired_size_col = self.roi_model['col']
        assert desired_size_row > img.shape[0] or desired_size_col > img.shape[1]
        delta_row = desired_size_row - img.shape[0]
        delta_col = desired_size_col - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_col // 2
        image = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top, pad_left, delta_col - pad_left,
                                   cv2.BORDER_CONSTANT, value=self.pad_value)
        return image

    def extract_freature_metric_learning(self, bbox, img, seg_mask, ind, normalize_type='MinMax_all'):
        min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
        img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb]
        msk_patch = seg_mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb] != ind
        assert normalize_type == 'MinMax_all'
        if normalize_type != 'MinMax_all':
            img_patch[msk_patch] = self.pad_value
        img_patch = img_patch.astype(np.float32)

        if normalize_type == 'regular':
            img = self.padding(img_patch) / self.max_img
        elif normalize_type == 'MinMax_all':
            assert img_patch.shape == (self.roi_model['row'], self.roi_model['col']), \
                f"Problem! {img_patch.shape} should be {(self.roi_model['row'], self.roi_model['col'])}"
            img_patch = (img_patch - self.min_cell) / (self.max_cell - self.min_cell)
            img = img_patch.squeeze()
        elif normalize_type == 'MinMaxCell':
            not_msk_patch = np.logical_not(msk_patch)
            img_patch[not_msk_patch] = (img_patch[not_msk_patch] - self.min_cell) / (self.max_cell - self.min_cell)
            img = self.padding(img_patch)
        else:
            assert False, "Not supported this type of normalization"
        assert img.min() >= 0 and img.max() <= 1, f"Problem! Image values are not in range! {[img.min(), img.max()]}"
        img = torch.from_numpy(img).float()
        with torch.no_grad():
            embedded_img = self.embedder(self.trunk(img[None, None, ...]))

        return embedded_img.numpy().squeeze()

    def preprocess_basic_features(self, path_to_write):
        cols = ["id", "seg_label",
                "frame_num",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "max_intensity", "mean_intensity", "min_intensity"
                ]
        for ind_data in range(self.__len__()):
            img, mask, im_path, mask_path = self[ind_data]

            im_num, mask_num = im_path.split(".")[-2][-3:], mask_path.split(".")[-2][-3:]
            assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            im_num_int = int(im_num)
            labels_mask = np.unique(mask)

            num_labels = labels_mask.shape[0]
            if 0 in labels_mask:
                num_labels = num_labels - 1
                flag_zero_label = True

            df = pd.DataFrame(index=range(num_labels), columns=cols)

            for ind, cell_id in enumerate(labels_mask):
                # Color 0 is assumed to be background or artifacts
                if cell_id == 0:
                    continue
                if flag_zero_label:
                    ind_df = ind - 1

                df.loc[ind_df, "id"] = cell_id

                # extracting statistics using regionprops
                properties = regionprops(np.uint8(mask == cell_id), img)[0]
                row, col = properties.centroid[0].round().astype(np.int16), \
                           properties.centroid[1].round().astype(np.int16)
                min_row, max_row, min_col, max_col = self.extract_patch(img, properties.bbox, col, row, cell_id)

                df.loc[ind_df, "seg_label"] = cell_id

                df.loc[ind_df, "min_row_bb"], df.loc[ind_df, "min_col_bb"], \
                df.loc[ind_df, "max_row_bb"], df.loc[ind_df, "max_col_bb"] = min_row, min_col, max_row, max_col

                df.loc[ind_df, "centroid_row"], df.loc[ind_df, "centroid_col"] = \
                    properties.centroid[0].round().astype(np.int16), \
                    properties.centroid[1].round().astype(np.int16)

                df.loc[ind_df, "max_intensity"], df.loc[ind_df, "mean_intensity"], df.loc[ind_df, "min_intensity"] = \
                    properties.max_intensity, properties.mean_intensity, properties.min_intensity

            df.loc[:, "frame_num"] = im_num_int

            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")

            sub_dir = op.join(self.path.split("/")[-2], self.sec_path)
            full_dir = op.join(path_to_write, sub_dir)
            full_dir = op.join(full_dir, "csv")
            os.makedirs(to_absolute_path(full_dir), exist_ok=True)
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            df.to_csv(to_absolute_path(file_path), index=False)
        print(f"files were saved to : {file_path}")

    def preprocess_features_metric_learning(self, path_to_write, dict_path):
        dict_params = torch.load(dict_path)

        self.min_cell = dict_params['min_all']
        self.max_cell = dict_params['max_all']
        self.roi_model = dict_params['roi']
        # models params
        model_name = dict_params['model_name']
        mlp_dims = dict_params['mlp_dims']
        mlp_normalized_features = dict_params['mlp_normalized_features']
        # models state_dict
        trunk_state_dict = dict_params['trunk_state_dict']
        embedder_state_dict = dict_params['embedder_state_dict']

        trunk = set_model_architecture(model_name)
        trunk.load_state_dict(trunk_state_dict)
        self.trunk = trunk
        self.trunk.eval()

        embedder = MLP(mlp_dims, normalized_feat=mlp_normalized_features)
        embedder.load_state_dict(embedder_state_dict)
        self.embedder = embedder
        self.embedder.eval()

        cols = ["id", "seg_label",
                "frame_num",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "max_intensity", "mean_intensity", "min_intensity"
                ]

        cols_resnet = [f'feat_{i}' for i in range(mlp_dims[-1])]
        cols += cols_resnet

        for ind_data in range(self.__len__()):
            img, mask, im_path, mask_path = self[ind_data]

            im_num, mask_num = im_path.split(".")[-2][-3:], mask_path.split(".")[-2][-3:]

            assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            im_num_int = int(im_num)
            labels_mask = np.unique(mask)

            num_labels = labels_mask.shape[0]
            if 0 in labels_mask:
                num_labels = num_labels - 1
                flag_zero_label = True

            df = pd.DataFrame(index=range(num_labels), columns=cols)

            for ind, cell_id in enumerate(labels_mask):
                # Color 0 is assumed to be background or artifacts
                if cell_id == 0:
                    continue
                if flag_zero_label:
                    ind_df = ind - 1

                df.loc[ind_df, "id"] = cell_id

                # extracting statistics using regionprops
                properties = regionprops(np.uint8(mask == cell_id), img)[0]
                row, col = properties.centroid[0].round().astype(np.int16), \
                           properties.centroid[1].round().astype(np.int16)
                min_row, max_row, min_col, max_col = self.extract_patch(img, properties.bbox, col, row, cell_id)
                bbox = (min_row, min_col, max_row, max_col)
                embedded_feat = self.extract_freature_metric_learning(bbox, img.copy(), mask.copy(), cell_id)
                df.loc[ind_df, cols_resnet] = embedded_feat
                df.loc[ind_df, "seg_label"] = cell_id

                df.loc[ind_df, "min_row_bb"], df.loc[ind_df, "min_col_bb"], \
                df.loc[ind_df, "max_row_bb"], df.loc[ind_df, "max_col_bb"] = min_row, min_col, max_row, max_col

                df.loc[ind_df, "centroid_row"], df.loc[ind_df, "centroid_col"] = \
                    properties.centroid[0].round().astype(np.int16), \
                    properties.centroid[1].round().astype(np.int16)

                df.loc[ind_df, "max_intensity"], df.loc[ind_df, "mean_intensity"], df.loc[ind_df, "min_intensity"] = \
                    properties.max_intensity, properties.mean_intensity, properties.min_intensity

            df.loc[:, "frame_num"] = im_num_int

            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")
            sub_dir = op.join(self.path.split("/")[-2], self.sec_path)
            full_dir = op.join(path_to_write, sub_dir)
            full_dir = op.join(full_dir, "csv")
            os.makedirs(to_absolute_path(full_dir), exist_ok=True)
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            df.to_csv(to_absolute_path(file_path), index=False)
        print(f"files were saved to : {file_path}")

def create_csv(input_images, input_masks,
               input_model, output_csv,
               input_seg, seg_dir,
               basic=False, sequences=['01', '02']
               ):
    dict_path = input_model
    path_output = output_csv
    for seq in sequences:
        curr_img_path = os.path.join(input_images, seq)
        curr_msk_path = os.path.join(input_masks, seq + "_GT/TRA")
        ds = TestDataset(
            debug=False,
            path=curr_img_path,
            path_marker=curr_msk_path,
            type_img="tif",
            sec_path=seq)
        if basic:
            ds.preprocess_basic_features(path_to_write=path_output)
        else:
            ds.preprocess_features_metric_learning(path_to_write=path_output,
                                                   dict_path=dict_path)

if __name__ == "__main__":
    create_csv_phase_contrast_resnet()
    create_csv_Huh7_resnet()



