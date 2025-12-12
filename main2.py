from pathlib import Path
import argparse
import yaml
import torch
from model import FBNETGEN
from newlearning import BasicTrain



def main(args,i):
    args = args
    with open(args.config_filename,encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.Loader)

        if config['model']['type'] == 'fbnetgen':
            model = FBNETGEN(config['model'], 200,
                             200, 100)

            use_train = BasicTrain


        train_process = use_train(
            config['train'], model, i)

        train_process.train3()
        train_process.heldout_test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_fbnetgen.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=5, type=int)

    args = parser.parse_args()
    for i in range(args.repeat_time):
        main(args,i)
