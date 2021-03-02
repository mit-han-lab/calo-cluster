import argparse

from hgcal_dev.utils.experiment import Experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('wandb_version')
    parser.add_argument('--ckpt_name')
    args = parser.parse_args()
    experiment = Experiment(args.wandb_version, args.ckpt_name)
    experiment.save_predictions()


if __name__ == '__main__':
    main()
