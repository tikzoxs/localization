#from simclr import SimCLR
from Eye_Crop import EyeCrop
import yaml
from data_utils_eye.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader) # This is a config file that consist of all the paramters neeed config the data loader.

    dataset = DataSetWrapper(config['batch_size'], **config['dataset']) #This is the dataloader object in pytorch.  Refer to data_aug/dataset_wrapper.py

    eyecrop = EyeCrop(dataset, config) #the model and stuff
    eyecrop.train()


if __name__ == "__main__":
    main()
