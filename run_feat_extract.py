import importlib
import dotenv
import hydra
from omegaconf import DictConfig
dotenv.load_dotenv(override=True)

@hydra.main(config_path="configs/feat_extract/", config_name="feat_extract.yaml")
def main(config: DictConfig):
    module = importlib.import_module(config._target_)
    module.create_csv(**dict(config.params))

if __name__ == "__main__":
    main()
