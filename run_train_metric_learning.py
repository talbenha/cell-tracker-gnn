from pytorch_metric_learning import losses, miners, reducers, distances, trainers, testers
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch
import tqdm
import pprint
from datetime import datetime

import importlib
import hydra
from omegaconf import DictConfig

from src_metric_learning.Data.sampler import MPerClassSampler_weighted
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

def train(device,
         patch_based,
         data_config,
         base_dir,
         model_name,
         embedding_dim,
         dataset_dict_keys,
         batch_size,
         num_epochs,
         num_workers,
         lr_trunk,
         lr_embedder,
         weight_decay,
         loss_function,
         loss_distance,
         loss_margin,
         loss_gamma,
         epsilon_miner,
         shorter,
         dataset_module,
         m_samples=4,
         avg_of_avgs=True,
         k="max_bin_count",
         test_interval=1,
         patience=20,
         normalized_feat=False,
         ):
    model_folder = os.path.join(base_dir, "saved_models")
    logs_folder = os.path.join(base_dir, "logs")
    tensorboard_folder = os.path.join(base_dir, "tensorboard")

    record_keeper, _, _ = logging_presets.get_record_keeper(logs_folder, tensorboard_folder)
    hooks = logging_presets.get_hook_container(record_keeper)
    if model_name == 'resnet18_3d':
        from src_metric_learning.modules.resnet_3d.resnet import set_model_architecture, MLP
    else:
        from src_metric_learning.modules.resnet_2d.resnet import set_model_architecture, MLP

    trunk = set_model_architecture(model_name)
    trunk_output_size = trunk.input_features_fc_layer
    trunk = torch.nn.DataParallel(trunk.to(device))
    embedder = torch.nn.DataParallel(MLP([trunk_output_size, embedding_dim],
                                         normalized_feat=normalized_feat).to(device))

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=lr_trunk, weight_decay=weight_decay)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=lr_embedder, weight_decay=weight_decay)

    train_dataset = dataset_module.ImgDataset(**data_config, type_data='train')
    val_dataset = dataset_module.ImgDataset(**data_config, type_data='valid')
    test_data = dataset_module.ImgDataset(**data_config, type_data='test')

    print(train_dataset.curr_roi)
    print(f"train_dataset length:{len(train_dataset)}")
    print(f"val_dataset length:{len(val_dataset)}")
    print(f"test_data length:{len(test_data)}")

    assert set(train_dataset.targets).isdisjoint(set(val_dataset.targets))

    # Set the loss function AND the mining function
    if loss_function == 'circle_loss':
        print("use circle_loss")
        loss = losses.CircleLoss(m=loss_margin, gamma=loss_gamma, distance=CosineSimilarity(), reducer=MeanReducer())
        miner = miners.MultiSimilarityMiner(epsilon=epsilon_miner)
    elif loss_function == 'MultiSimilarityLoss':
        print("use MultiSimilarityLoss")
        loss = losses.MultiSimilarityLoss(distance=CosineSimilarity(), reducer=MeanReducer())
        miner = miners.MultiSimilarityMiner(epsilon=epsilon_miner)
    elif loss_function == 'triplet_loss':
        print("use triplet_loss")
        if loss_distance == 'CosineSimilarity':
            distance = distances.CosineSimilarity()
        elif loss_distance == 'LpDistance':
            distance = distances.LpDistance()
        else:
            assert False
        reducer = reducers.ThresholdReducer(low=0)
        loss = losses.TripletMarginLoss(margin=loss_margin, distance=distance, reducer=reducer)
        miner = miners.TripletMarginMiner(margin=loss_margin, distance=distance, type_of_triplets="semihard")
    else:
        assert False
    # Set the dataloader sampler
    sampler = MPerClassSampler_weighted(train_dataset.targets, frames=train_dataset.frames_for_sampler, m=m_samples,
                                        length_before_new_iter=len(train_dataset))

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}
    dataset_dict = {}

    for key in dataset_dict_keys:
        if key == 'train':
            dataset_dict['train'] = train_dataset
        if key == 'val':
            dataset_dict['val'] = val_dataset
        if key == 'test':
            dataset_dict['test'] = test_data

    print(f"dataset_dict keys: {dataset_dict.keys()}")

    accuracy_calculator = AccuracyCalculator(avg_of_avgs=avg_of_avgs, k=k)

    # Create the tester

    def end_of_testing_hook(tester):
        for split, (embeddings, labels) in tester.embeddings_and_labels.items():
            dataset = common_functions.EmbeddingDataset(embeddings.cpu().numpy(), labels.squeeze(1).cpu().numpy())
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=1)
            total_loss = 0
            with torch.no_grad():
                print(f"getting loss for {split} set")
                for E, L in tqdm.tqdm(dataloader):
                    total_loss += loss(E, L)
            total_loss /= len(dataloader)
            tester.all_accuracies[split]["loss"] = total_loss
        hooks.end_of_testing_hook(tester)

    if shorter:
        end_of_testing_hook_for_tester = hooks.end_of_testing_hook
    else:
        end_of_testing_hook_for_tester = end_of_testing_hook

    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=end_of_testing_hook_for_tester,
                                                dataloader_num_workers=num_workers,
                                                accuracy_calculator=accuracy_calculator
                                                )

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                                dataset_dict,
                                                model_folder)
#                                                 test_interval=test_interval,
#                                                 patience=patience)

    trainer = trainers.MetricLossOnly(models,
                                      optimizers,
                                      batch_size,
                                      loss_funcs,
                                      mining_funcs,
                                      train_dataset,
                                      sampler=sampler,
                                      dataloader_num_workers=num_workers,
                                      end_of_iteration_hook=hooks.end_of_iteration_hook,
                                      end_of_epoch_hook=end_of_epoch_hook)

    trainer.train(num_epochs=num_epochs)

    work_dir = os.getcwd()

    save_model = os.path.join(work_dir, base_dir, "saved_models")
    for file in os.listdir(save_model):
        if file.startswith('trunk_best'):
            trunk_ckpt_path = os.path.join(save_model, file)
        if file.startswith('embedder_best'):
            embedder_ckpt_path = os.path.join(save_model, file)

    print(f"best trunk_ckpt: {trunk_ckpt_path}")
    print(f"best embedder_ckpt: {embedder_ckpt_path}")
    trunk_ckpt = torch.load(trunk_ckpt_path)
    embedder_ckpt = torch.load(embedder_ckpt_path)

    dict_params = {}
    if patch_based:
        assert val_dataset.min_all == train_dataset.min_all
        assert val_dataset.max_all == train_dataset.max_all
        dict_params['min_all'] = val_dataset.min_all
        dict_params['max_all'] = val_dataset.max_all
    else:

        dict_params['min_cell'] = test_data.min_cell
        dict_params['max_cell'] = test_data.max_cell
        dict_params['pad_value'] = test_data.pad_value

    dict_params['roi'] = test_data.curr_roi

    # models params
    dict_params['model_name'] = model_name
    dict_params['mlp_dims'] = [trunk_output_size, embedding_dim]
    dict_params['mlp_normalized_features'] = normalized_feat

    # models state_dict
    dict_params['trunk_state_dict'] = trunk_ckpt
    dict_params['embedder_state_dict'] = embedder_ckpt

    save_path = os.path.join(work_dir, 'all_params.pth')
    torch.save(dict_params, 'all_params.pth')
    print(f'save: {save_path}')


def run(config, data_config, dataset_module, exp_name, patch_based):
    datetime_object = str(datetime.now())
    datetime_object = datetime_object.split('.')[0].replace(':', '-').replace(' ', '/')
    print(f"start time: {datetime_object}")
    base_dir = "logs_" + exp_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Config dictionary")
    pprint.pprint(config, sort_dicts=False)

    print("data_config dictionary")
    pprint.pprint(data_config, sort_dicts=False)

    train(device,
          patch_based=patch_based,
          dataset_module=dataset_module,
          base_dir=base_dir,
          data_config=data_config,
          **config)


@hydra.main(config_path="configs/metric_learning/", config_name="config_2D.yaml")
def main(config: DictConfig):
    dict_kwargs = config.kwargs
    data_config = config.dataset.kwargs
    target_dataset = config.dataset._target_
    flag = 'MinMax_all' in data_config.normalize_type
    dataset_module = importlib.import_module(target_dataset)
    run(dict(dict_kwargs), dict(data_config), dataset_module, config.exp_name, patch_based=flag)

if __name__ == "__main__":
    main()

