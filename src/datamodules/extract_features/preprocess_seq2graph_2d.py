import os
import os.path as op
import shutil
from PIL import Image
import torch
import numpy as np
import pandas as pd
import cv2
import warnings
from torch.utils.data import Dataset
from skimage.measure import regionprops
from hydra.utils import get_original_cwd, to_absolute_path

from src_metric_learning.modules.resnet_2d.resnet import set_model_architecture, MLP


class TestDataset(Dataset):

    def __init__(self,
                 path: str,
                 path_masks: str,
                 path_result: str,
                 type_img: str,
                 sec_path
                 ):
        path = os.path.join(get_original_cwd(), path) if path.startswith('./') else path
        path_masks = os.path.join(get_original_cwd(), path_masks) if path_masks.startswith('./') else path_masks
        path_result = os.path.join(get_original_cwd(), path_result) if path_result.startswith('./') else path_result

        self.path = path
        self.sec_path = sec_path
        self.path_result = path_result
        type_masks = type_img
        dir_img = path
        dir_masks = path_masks
        dir_results = path_result

        assert os.path.exists(dir_img), f"Image paths ({dir_img}) is not exist, please fix it!"
        assert os.path.exists(dir_masks), f"Masks paths ({dir_masks}) is not exist, please fix it!"
        assert os.path.exists(dir_results), f"Result paths ({dir_results}) is not exist, please fix it!"
        self.images = []
        if os.path.exists(dir_img):
            self.images = [os.path.join(dir_img, fname) for fname in sorted(os.listdir(dir_img))
                           if type_img in fname]
        self.masks = []
        if os.path.exists(dir_masks):
            self.masks = [os.path.join(dir_masks, fname) for fname in sorted(os.listdir(dir_masks))
                          if type_masks in fname]

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

        mask_path, mask = None, None
        if len(self.masks):
            mask_path = self.masks[idx]
            mask = np.array(Image.open(mask_path))

        result_path, result = None, None
        if len(self.results):
            result_path = self.results[idx]
            result = np.array(Image.open(result_path))
        flag = True
        if im_path is not None:
            flag = False
            im_num = im_path.split(".")[-2][-3:]
        if mask_path is not None:
            flag = False
            mask_num = mask_path.split(".")[-2][-3:]
        if result_path is not None:
            flag = False
            result_num = result_path.split(".")[-2][-3:]

        if flag:
            assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

        return image, mask, result, im_path, mask_path, result_path

    def __len__(self):
        return len(self.images)

    def padding(self, img):
        desired_size_row = self.roi_model['row']
        desired_size_col = self.roi_model['col']
        assert desired_size_row > img.shape[0] or desired_size_col > img.shape[1], "the patch dimension is bigger than image shape"
        delta_row = desired_size_row - img.shape[0]
        delta_col = desired_size_col - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_col // 2
        image = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top, pad_left, delta_col - pad_left,
                                   cv2.BORDER_CONSTANT, value=self.pad_value)
        return image

    def extract_freature_metric_learning(self, bbox, img, seg_mask, ind, normalize_type='MinMaxCell'):
        min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
        img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb]
        msk_patch = seg_mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb] != ind
        img_patch[msk_patch] = self.pad_value
        img_patch = img_patch.astype(np.float32)

        if normalize_type == 'regular':
            img = self.padding(img_patch) / self.max_img
        elif normalize_type == 'MinMaxCell':
            not_msk_patch = np.logical_not(msk_patch)
            img_patch[not_msk_patch] = (img_patch[not_msk_patch] - self.min_cell) / (self.max_cell - self.min_cell)
            img = self.padding(img_patch)
        else:
            assert False, "Not supported this type of normalization"

        img = torch.from_numpy(img).float()
        with torch.no_grad():
            embedded_img = self.embedder(self.trunk(img[None, None, ...]))

        return embedded_img.numpy().squeeze()

    def preprocess_basic_features(self, path_to_write):
        cols = ["id",
                "frame_num",
                "area",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "major_axis_length", "minor_axis_length",
                "max_intensity", "mean_intensity", "min_intensity"
                ]

        for ind_data in range(self.__len__()):
            img, mask, result, im_path, mask_path, result_path = self[ind_data]

            im_num, mask_num = im_path.split(".")[-2][-3:], mask_path.split(".")[-2][-3:]
            result_num = result_path.split(".")[-2][-3:]

            if 'hela' in im_path.lower() and 'Silver_GT' in result_path:
                if not np.all(np.unique(mask) == np.unique(result)):
                    warnings.warn("Not as expected for Hela! Silver GT is inconsistent with TRA")

            assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

            num_labels = np.unique(result).shape[0] - 1

            df = pd.DataFrame(index=range(num_labels), columns=cols)

            for ind, id_res in enumerate(np.unique(result)):
                # Color 0 is assumed to be background or artifacts
                row_ind = ind-1
                if id_res == 0:
                    continue

                # compute the largest overlapped cell
                curr_result_by_id = np.uint8(result == id_res).copy()
                cpy_mask = mask.copy()

                res_mult_mask = np.multiply(curr_result_by_id, cpy_mask)
                res_mult_mask_bin = np.uint8(res_mult_mask > 0)
                res_label = np.argmax(np.bincount(res_mult_mask.flat, weights=res_mult_mask_bin.flat))

                if 'hela' in im_path.lower() and 'Silver_GT' in result_path:
                    if not (res_label == id_res):
                        warnings.warn("Not as expected for Hela! curr Silver GT object is inconsistent with TRA object")

                if res_label == 0:
                    warnings.warn(f"Pay Attention! there is no result for {id_res}!")
                    df = df.drop([row_ind])
                    continue

                # extracting statistics using regionprops
                properties = regionprops(np.uint8(result == id_res), img)[0]

                df.loc[row_ind, "id"] = res_label
                df.loc[row_ind, "area"] = properties.area

                df.loc[row_ind, "min_row_bb"], df.loc[row_ind, "min_col_bb"], \
                df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox

                df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
                    properties.centroid[0].round().astype(np.int16), \
                    properties.centroid[1].round().astype(np.int16)

                df.loc[row_ind, "major_axis_length"], df.loc[row_ind, "minor_axis_length"] = \
                    properties.major_axis_length, properties.minor_axis_length

                df.loc[row_ind, "max_intensity"], df.loc[row_ind, "mean_intensity"], df.loc[row_ind, "min_intensity"] = \
                    properties.max_intensity, properties.mean_intensity, properties.min_intensity

            df.loc[:, "frame_num"] = int(im_num)

            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")

            sub_dir = op.join(self.path.split("/")[-2], self.sec_path)
            full_dir = op.join(path_to_write, sub_dir)
            full_dir = op.join(full_dir, "csv")
            os.makedirs(to_absolute_path(full_dir), exist_ok=True)
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            df.to_csv(to_absolute_path(file_path), index=False)
        print(f"files were saved to : {full_dir}")

    def preprocess_features_metric_learning(self, path_to_write, dict_path):
        dict_params = torch.load(dict_path)

        self.min_cell = dict_params['min_cell'][int(self.sec_path) - 1]
        self.max_cell = dict_params['max_cell'][int(self.sec_path) - 1]
        self.roi_model = dict_params['roi']
        self.pad_value = dict_params['pad_value']
        # models params
        model_name = dict_params['model_name']
        mlp_dims = dict_params['mlp_dims']
        mlp_normalized_features = dict_params['mlp_normalized_features']
        # models state_dict
        trunk_state_dict = dict_params['trunk_state_dict']
        embedder_state_dict = dict_params['embedder_state_dict']

        trunk = set_model_architecture(model_name)
        trunk.load_state_dict(trunk_state_dict)
        # trunk = torch.nn.DataParallel(trunk) # .to("cpu")
        self.trunk = trunk
        self.trunk.eval()

        embedder = MLP(mlp_dims, normalized_feat=mlp_normalized_features) #.to("cpu")
        embedder.load_state_dict(embedder_state_dict)
        # embedder = torch.nn.DataParallel(embedder)
        self.embedder = embedder
        self.embedder.eval()

        cols = ["id", "seg_label",
                "frame_num",
                "area",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "major_axis_length", "minor_axis_length",
                "max_intensity", "mean_intensity", "min_intensity"
                ]

        cols_resnet = [f'feat_{i}' for i in range(mlp_dims[-1])]
        cols += cols_resnet

        for ind_data in range(self.__len__()):
            img, mask, result, im_path, mask_path, result_path = self[ind_data]

            im_num, mask_num = im_path.split(".")[-2][-3:], mask_path.split(".")[-2][-3:]
            result_num = result_path.split(".")[-2][-3:]

            if 'hela' in im_path.lower() and 'Silver_GT' in result_path:
                if not np.all(np.unique(mask) == np.unique(result)):
                    warnings.warn("Not as expected for Hela! Silver GT is inconsistent with TRA")

            assert im_num == mask_num, f"Image number ({im_num}) is not equal to mask number ({mask_num})"
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

            num_labels = np.unique(result).shape[0] - 1

            df = pd.DataFrame(index=range(num_labels), columns=cols)

            for ind, id_res in enumerate(np.unique(result)):
                # Color 0 is assumed to be background or artifacts
                row_ind = ind-1
                if id_res == 0:
                    continue

                # compute the largest overlapped cell
                curr_result_by_id = np.uint8(result == id_res).copy()
                cpy_mask = mask.copy()

                res_mult_mask = np.multiply(curr_result_by_id, cpy_mask)
                res_mult_mask_bin = np.uint8(res_mult_mask > 0)
                res_label = np.argmax(np.bincount(res_mult_mask.flat, weights=res_mult_mask_bin.flat))

                if 'hela' in im_path.lower() and 'Silver_GT' in result_path:
                    if not (res_label == id_res):
                        warnings.warn("Not as expected for Hela! curr Silver GT object is inconsistent with TRA object")

                if res_label == 0:
                    warnings.warn(f"Pay Attention! there is no result for {id_res}!")
                    df = df.drop([row_ind])
                    continue

                # extracting statistics using regionprops
                properties = regionprops(np.uint8(result == id_res), img)[0]

                embedded_feat = self.extract_freature_metric_learning(properties.bbox, img.copy(), result.copy(), id_res)
                df.loc[row_ind, cols_resnet] = embedded_feat
                df.loc[row_ind, "seg_label"] = id_res

                df.loc[row_ind, "id"] = res_label
                df.loc[row_ind, "area"] = properties.area


                df.loc[row_ind, "min_row_bb"], df.loc[row_ind, "min_col_bb"], \
                df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox

                df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
                    properties.centroid[0].round().astype(np.int16), \
                    properties.centroid[1].round().astype(np.int16)

                df.loc[row_ind, "major_axis_length"], df.loc[row_ind, "minor_axis_length"] = \
                    properties.major_axis_length, properties.minor_axis_length

                df.loc[row_ind, "max_intensity"], df.loc[row_ind, "mean_intensity"], df.loc[row_ind, "min_intensity"] = \
                    properties.max_intensity, properties.mean_intensity, properties.min_intensity

            df.loc[:, "frame_num"] = int(im_num)

            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")


            sub_dir = op.join(self.path.split("/")[-2], self.sec_path)
            full_dir = op.join(path_to_write, sub_dir)
            full_dir = op.join(full_dir, "csv")
            os.makedirs(to_absolute_path(full_dir), exist_ok=True)
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            df.to_csv(to_absolute_path(file_path), index=False)
        print(f"files were saved to : {full_dir}")


def create_csv(input_images, input_masks, input_seg,
               input_model, output_csv, basic=False,
               sequences=['01', '02'], seg_dir='_ST/SEG',
               ):
    dict_path = input_model
    path_output = output_csv
    path_Seg_result = input_seg
    for seq in sequences:
        curr_img_path = os.path.join(input_images, seq)
        curr_msk_path = os.path.join(input_masks, seq + "_GT/TRA")
        curr_seg_path = os.path.join(path_Seg_result, seq + seg_dir)
        ds = TestDataset(
            path=curr_img_path,
            path_masks=curr_msk_path,
            path_result=curr_seg_path,
            type_img="tif",
            sec_path=seq)
        if basic:
            ds.preprocess_basic_features(path_to_write=path_output)
        else:
            ds.preprocess_features_metric_learning(path_to_write=path_output, dict_path=dict_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ii', type=str, required=True, help='input images directory')
    parser.add_argument('-imsk', type=str, required=True, help='input TRA masks directory')
    parser.add_argument('-iseg', type=str, required=True, help='input segmentation directory')
    parser.add_argument('-im', type=str, required=True, help='metric learning model params directory')
    parser.add_argument('-sd', type=str, default=None, help='segmentation directory name ')
    parser.add_argument('-seq', type=str, default=None, nargs="*", help='sequences list of strings')

    parser.add_argument('-oc', type=str, required=True, help='output csv directory')

    args = parser.parse_args()

    input_images = args.ii
    input_masks = args.imsk
    input_segmentation = args.iseg
    input_model = args.im
    output_csv = args.oc
    seg_dir = args.sd
    sequences = args.seq

    create_csv(input_images, input_segmentation, input_model, output_csv, sequences, seg_dir)

