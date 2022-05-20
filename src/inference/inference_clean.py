import os
import yaml
import torch

from src.models.celltrack_plmodel import CellTrackLitModel
from graph_dataset_inference import CellTrackDataset
import warnings
warnings.filterwarnings("ignore")


def predict(ckpt_path, path_csv_output, num_seq):
    """Inference with trained model.
    It loads trained model from checkpoint.
    Then it creates graph and make prediction.
    """

    CKPT_PATH = ckpt_path
    path_output = path_csv_output

    folder_path = CKPT_PATH
    for i in range(2):
        folder_path = folder_path[:folder_path.rfind('/')]

    config_path = os.path.join(folder_path, '.hydra/config.yaml')
    config = yaml.load(open(config_path))

    print(f"load model from: {CKPT_PATH}")
    data_yaml = config['datamodule']

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = CellTrackLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    data_yaml['dataset_params']['num_frames'] = 'all'
    data_yaml['dataset_params']['main_path'] = path_output

    second_path = num_seq
    data_yaml['dataset_params']['dirs_path']['test'] = [second_path + "_CSV"]

    data_train: CellTrackDataset = CellTrackDataset(**data_yaml['dataset_params'], split='test')
    data_list, df_list = data_train.all_data['test']
    test_data, df_data = data_list[0], df_list[0]
    x, x2, edge_index, edge_feature = test_data.x, test_data.x_2, test_data.edge_index, test_data.edge_feat

    outputs = trained_model((x, x2), edge_index, edge_feature.float())
    data_path = os.path.join(path_output, second_path) + '_RES_inference'
    path_output_folder = data_path
    print(f"save path : {path_output_folder}")
    os.makedirs(path_output_folder, exist_ok=True)
    file1 = os.path.join(path_output_folder, 'pytorch_geometric_data.pt')
    file2 = os.path.join(path_output_folder, 'all_data_df.csv')
    file3 = os.path.join(path_output_folder, 'raw_output.pt')
    print(f"Save inference files: \n - {file1} \n - {file2} \n - {file3}")
    df_data.to_csv(file2)
    torch.save(test_data, file1)
    torch.save(outputs, file3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', type=str, required=True, help='model params full path')

    parser.add_argument('-ns', type=str, required=True, help='number of sequence - string 01/02')
    parser.add_argument('-oc', type=str, required=True, help='output csv directory')

    args = parser.parse_args()

    model_path = args.mp
    num_seq = args.ns
    output_csv = args.oc
    assert num_seq == '01' or num_seq == '02'
    predict(model_path, output_csv, num_seq)
