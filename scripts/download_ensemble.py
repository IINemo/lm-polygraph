import argparse

from transformers import AutoModel, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', action='append', required=True, help='Models that are to be used in ensemble')
    parser.add_argument('-s', '--save', required=True, help='Folder to save models')
    return parser.parse_args()


def main():
    args = parse_args()
    for model in args.model:
        AutoModel.from_pretrained(model).save_pretrained(f'{args.save}/{model.replace("/", "_")}')
        AutoTokenizer.from_pretrained(model).save_pretrained(f'{args.save}/{model.replace("/", "_")}')


if __name__ == '__main__':
    main()
