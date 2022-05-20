import os
import os.path as op

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops
from skimage.morphology import label
import warnings
from src_metric_learning.modules.resnet_2d.resnet import set_model_architecture, MLP

import torch

class TestDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self,
                 path: str,

                 path_result: str,

                 type_img: str,
                 type_masks: str):

        self.path = path

        self.path_result = path_result

        dir_img = path
        dir_results = path_result

        self.images = []
        if os.path.exists(dir_img):
            self.images = [os.path.join(dir_img, fname) for fname in sorted(os.listdir(dir_img))
                           if type_img in fname]

        self.results = []
        if os.path.exists(dir_results):
            self.results = [os.path.join(dir_results, fname) for fname in sorted(os.listdir(dir_results))
                            if type_masks in fname]

    def __getitem__(self, idx):
        assert len(self.images) or len(self.images), "both directories are empty, please fix it!"

        im_path, image = None, None
        if len(self.images):
            im_path = self.images[idx]
            image = np.array(Image.open(im_path))

        result_path, result = None, None
        if len(self.results):
            result_path = self.results[idx]
            result = np.array(Image.open(result_path))
        flag = True
        if im_path is not None:
            flag = False
            im_num = im_path.split(".")[-2][-3:]

        if result_path is not None:
            flag = False
            result_num = result_path.split(".")[-2][-3:]

        if flag:
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

        return image, result, im_path, result_path

    def __len__(self):
        return len(self.images)

    def extract_patch(self, img, bbox):
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
        return min_row, max_row, min_col, max_col

    def extract_freature_metric_learning(self, bbox, img, seg_mask, ind, normalize_type='MinMax_all'):
        min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
        img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb]
        msk_patch = seg_mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb] != ind
        assert normalize_type == 'MinMax_all'
        if normalize_type != 'MinMax_all':
            img_patch[msk_patch] = self.pad_value
        img_patch = img_patch.astype(np.float32)

        if normalize_type == 'MinMax_all':
            assert img_patch.shape == (self.roi_model['row'], self.roi_model['col']), \
                f"Problem! {img_patch.shape} should be {(self.roi_model['row'], self.roi_model['col'])}"
            img_patch = (img_patch - self.min_cell) / (self.max_cell - self.min_cell)
            img = img_patch.squeeze()
        else:
            assert False, "Not supported this type of normalization"
        assert img.min() >= 0 and img.max() <= 1, "Problem! Image values are not in range!"
        img = torch.from_numpy(img).float()
        with torch.no_grad():
            embedded_img = self.embedder(self.trunk(img[None, None, ...]))

        return embedded_img.numpy().squeeze()

    def correct_masks(self, min_cell_size):
        n_changes = 0
        for ind_data in range(self.__len__()):
            per_cell_change = False
            per_mask_change = False

            img, result, im_path, result_path = self[ind_data]
            res_save = result.copy()
            print(f"start: {result_path}")
            labels_mask = result.copy()
            while True:
                bin_mask = labels_mask > 0
                re_label_mask = label(bin_mask)
                un_labels, counts = np.unique(re_label_mask, return_counts=True)

                if np.any(counts < min_cell_size):
                    per_mask_change = True

                    # print(f"{im_path}: \n {counts}")
                    first_label_ind = np.argwhere(counts < min_cell_size)
                    if first_label_ind.size > 1:
                        first_label_ind = first_label_ind.squeeze()[0]
                    first_label_num = un_labels[first_label_ind]
                    labels_mask[re_label_mask == first_label_num] = 0
                else:
                    break
            bin_mask = (labels_mask > 0) * 1.0
            result = np.multiply(result, bin_mask)
            if not np.all(np.unique(result) == np.unique(res_save)):
                warnings.warn(
                    f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")

            # assert np.all(np.unique(result) == np.unique(res_save))
            for ind, id_res in enumerate(np.unique(result)):
                if id_res == 0:
                    continue
                bin_mask = (result == id_res).copy()
                while True:
                    re_label_mask = label(bin_mask)
                    un_labels, counts = np.unique(re_label_mask, return_counts=True)

                    if np.any(counts < min_cell_size):
                        per_cell_change = True
                        # print(f"{im_path}: \n {counts}")
                        first_label_ind = np.argwhere(counts < min_cell_size)
                        if first_label_ind.size > 1:
                            first_label_ind = first_label_ind.squeeze()[0]
                        first_label_num = un_labels[first_label_ind]
                        curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                        bin_mask[curr_mask] = False
                        result[curr_mask] = 0.0
                    else:
                        break
                while True:
                    re_label_mask = label(bin_mask)
                    un_labels, counts = np.unique(re_label_mask, return_counts=True)
                    if un_labels.shape[0] > 2:
                        per_cell_change = True
                        n_changes += 1
                        # print(f"un_labels.shape[0] > 2 : {im_path}: \n {counts}")
                        first_label_ind = np.argmin(counts)
                        if first_label_ind.size > 1:
                            first_label_ind = first_label_ind.squeeze()[0]
                        first_label_num = un_labels[first_label_ind]
                        curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                        bin_mask[curr_mask] = False
                        result[curr_mask] = 0.0
                    else:
                        break
            if not np.all(np.unique(result) == np.unique(res_save)):
                warnings.warn(
                    f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")
            if per_cell_change or per_mask_change:
                n_changes += 1
                res1 = (res_save > 0) * 1.0
                res2 = (result > 0) * 1.0
                n_pixels = np.abs(res1 - res2).sum()
                print(f"per_mask_change={per_mask_change}, per_cell_change={per_cell_change}, number of changed pixels: {n_pixels}")

                io.imsave(result_path, result.astype(np.uint16), compress=6)

        print(f"number of detected changes: {n_changes}")


    def preprocess_features_w_metric_learning(self, path_to_write, dict_path):
        dict_params = torch.load(dict_path)
        kernel = (64, 64)
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
            img, result, im_path, result_path = self[ind_data]
            mask_path = result_path
            mask = result
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

                centroid_row, centroid_col = properties.centroid[0].round().astype(np.int16), \
                                             properties.centroid[1].round().astype(np.int16)

                min_row_bb, min_col_bb, max_row_bb, max_col_bb = properties.bbox
                if max_row_bb - min_row_bb < kernel[0]:
                    if max_col_bb - min_col_bb < kernel[1]:
                        min_row, max_row, min_col, max_col = self.extract_patch(img, properties.bbox)
                    else:
                        min_col = max(centroid_col - 1, 0)
                        max_col = min(centroid_col + 1, mask.shape[1] - 1)
                        min_row, max_row, min_col, max_col = self.extract_patch(img, (min_row_bb, min_col, max_row_bb, max_col))
                else:
                    if max_col_bb - min_col_bb < kernel[1]:
                        min_row = max(centroid_row - 1, 0)
                        max_row = min(centroid_row + 1, mask.shape[0] - 1)
                        min_row, max_row, min_col, max_col = self.extract_patch(img, (min_row, min_col_bb, max_row, max_col_bb))
                    else:
                        min_row = max(centroid_row - 1, 0)
                        max_row = min(centroid_row + 1, mask.shape[0] - 1)
                        min_col = max(centroid_col - 1, 0)
                        max_col = min(centroid_col + 1, mask.shape[1] - 1)
                        min_row, max_row, min_col, max_col = self.extract_patch(img, (min_row, min_col, max_row, max_col))

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

            full_dir = op.join(path_to_write, "csv")
            os.makedirs(full_dir, exist_ok=True)
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            print(f"save file to : {file_path}")
            df.to_csv(file_path, index=False)


def create_csv(input_images, input_seg, input_model, output_csv, min_cell_size):
    dict_path = input_model
    path_output = output_csv
    path_Seg_result = input_seg
    ds = TestDataset(
        path=input_images,
        path_result=path_Seg_result,
        type_img="tif",
        type_masks="tif")
    ds.correct_masks(min_cell_size)
    ds.preprocess_features_w_metric_learning(path_to_write=path_output, dict_path=dict_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ii', type=str, required=True, help='input images directory')
    parser.add_argument('-iseg', type=str, required=True, help='input segmentation directory')
    parser.add_argument('-im', type=str, required=True, help='metric learning model params directory')
    parser.add_argument('-cs', type=int, required=True, help='min cell size')

    parser.add_argument('-oc', type=str, required=True, help='output csv directory')

    args = parser.parse_args()

    min_cell_size = args.cs
    input_images = args.ii
    input_segmentation = args.iseg
    input_model = args.im

    output_csv = args.oc

    create_csv(input_images, input_segmentation, input_model, output_csv, min_cell_size)


