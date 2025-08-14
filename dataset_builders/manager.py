import argparse
import os
import pathlib
import tempfile


import huggingface_hub


from build_dataset import DATASET_CONFIG, build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=DATASET_CONFIG.keys(),
        required=True,
        help="Dataset to build",
    )
    parser.add_argument(
        "-p",
        "--publish",
        action="store_true",
        help="Should dataset be published to HF (requires HF_TOKEN)",
    )
    parser.add_argument(
        "-n",
        "--namespace",
        default="LM-Polygraph",
        help="Namespace, where dataset will be located (LM-Polygraph by default)",
    )
    parser.add_argument(
        "-s",
        "--save-to-disk",
        action="store_true",
        help="Should dataset be saved to disk (useful for testing)",
    )

    return parser.parse_args()


def get_dataset_path(dataset_name, namespace):
    config = DATASET_CONFIG[dataset_name]
    return f"{namespace}/{config['dataset']}", config["subset"]


def create_readme_file(dataset_name, namespace):
    with open("template_readme.md") as f:
        template_readme = f.read()
    dataset = DATASET_CONFIG[dataset_name]["dataset"]
    src_url = f"https://huggingface.co/datasets/{dataset}"

    readme_contents = template_readme.format(
        empty_brackets="{}",
        dataset=dataset,
        src_url=src_url,
    )

    readme_file = tempfile.mktemp()
    with open(readme_file, "w") as f:
        f.write(readme_contents)

    return readme_file


def create_repo_if_not_exists(dataset_name, namespace):
    path, _ = get_dataset_path(dataset_name, namespace)
    api = huggingface_hub.HfApi()
    try:
        api.create_repo(path, repo_type="dataset")
    except huggingface_hub.errors.HfHubHTTPError:
        # We should not reupload files, since rewriting README.md erases dataset metadata, and subsets will not work
        return

    readme_file = create_readme_file(dataset_name, namespace)
    print("Uploading README.md")
    api.upload_file(
        path_or_fileobj=readme_file,
        path_in_repo="README.md",
        repo_id=path,
        repo_type="dataset",
    )


def main():
    args = parse_args()
    if args.publish:
        assert os.environ.get(
            "HF_TOKEN"
        ), "Please set HF_TOKEN in order to publish dataset"
    if not args.save_to_disk and not args.publish:
        print(
            "WARNING: --publish and --save-to-disk are not specified, so dataset will not be stored anywhere"
        )
    dataset = build_dataset(args.dataset)
    if args.save_to_disk:
        dataset.save_to_disk(pathlib.Path("result") / args.dataset)

    if args.publish:
        create_repo_if_not_exists(args.dataset, args.namespace)
        path, subset = get_dataset_path(args.dataset, args.namespace)
        print(f"Publishing dataset to path {path} with subset {subset}")
        dataset.push_to_hub(
            path,
            subset,
            set_default=False,
            token=os.environ["HF_TOKEN"],
            max_shard_size="1GB",  # Limit shard size to avoid issues with large datasets
        )


if __name__ == "__main__":
    main()
