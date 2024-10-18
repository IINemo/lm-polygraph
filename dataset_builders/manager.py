import argparse
import os
import subprocess


from build_dataset import DATASET_CONFIG, build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASET_CONFIG.keys(), required=True, help='Dataset to build, or "all" for every dataset')
    parser.add_argument('-p', '--publish', action='store_true', help='Should dataset be published to HF (requires HF_TOKEN)')
    parser.add_argument('-n', '--namespace', default='LM-Polygraph', help='Namespace, where dataset will be located (LM-Polygraph by default)')
    parser.add_argument('-s', '--save-to-disk', action='store_true', help='Should dataset be saved to disk in CSV format (useful for testing)')

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = build_dataset(args.dataset)
    if args.save_to_disk:
        os.mkdir(args.dataset)
        dataset['train'].to_csv(f'{args.dataset}/train.csv')
        dataset['test'].to_csv(f'{args.dataset}/test.csv')
    if args.publish:
        assert os.environ.get("HF_TOKEN"), "Please set HF_TOKEN in order to publish dataset"
        dataset.push_to_hub(f'{args.namespace}/{args.dataset}', token=os.environ["HF_TOKEN"])
        # or
        # dataset.push_to_hub(f'{args.namespace}/polygraph', args.dataset, token=os.environ["HF_TOKEN"])




if __name__ == '__main__':
    main()