import argparse

import yaml


def parse_args():
    ap = argparse.ArgumentParser("Binary NNs")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="vgg19")
    ap.add_argument("--binary-weights", action="store_true")
    ap.add_argument("--binary-act", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    config_path = args.config
    with open(config_path, mode="r") as file:
        config = yaml.safe_load(file)

    print(config)


if __name__ == '__main__':
    main()

    print("end of program")
