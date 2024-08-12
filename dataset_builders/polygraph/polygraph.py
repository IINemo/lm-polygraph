import datasets


VERSION = datasets.Version("0.0.1")


DATASET_CONFIG = {
    "xsum": {
        "name": "xsum",
        "splits": [
            "train",
            "validation",
            "test"
        ],
        "input_column": "document",
        "output_column": "summary",
        "prompt": "Here's the text and it's short one-sentence summary.\n\nText:\n{text}\n\nSummary (one sentence):\n"
    },
    "aeslc": {
        "name": "aeslc",
        "splits": [
            "train",
            "validation",
            "test"
        ],
        "input_column": "email_body",
        "output_column": "subject_line",
        "prompt": "Write a short subject line for the email. Output only the subject line itself.\n\nEmail:\n{text}\n\nSubject line:\n"
    },
    "trivia_qa_tiny": {
        "name": "SpeedOfMagic/trivia_qa_tiny",
        "splits": [
            "train",
            "test"
        ],
        "input_column": "question",
        "output_column": "answer"
    }
}


class PolygraphConfig(datasets.BuilderConfig):
    """BuilderConfig for xsum"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Polygraph(datasets.GeneratorBasedBuilder):
    """lm-polygraph wrapper for xsum dataset"""

    BUILDER_CONFIG_CLASS = PolygraphConfig
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="xsum", version=VERSION, description="Dataset xsum, processed by lm-polygraph"),
        datasets.BuilderConfig(name="aeslc", version=VERSION, description="Dataset aeslc, processed by lm-polygraph"),
        datasets.BuilderConfig(name="trivia_qa_tiny", version=VERSION, description="Dataset SpeedOfMagic/trivia_qa_tiny, processed by lm-polygraph"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="lm-polygraph wrapper for datasets",
            features=datasets.Features(
                {
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                }
            ),
        )

    def _prepare_dataset(self, dataset, input_column, output_column, prompt):
        x, y = dataset[input_column], dataset[output_column]
        if prompt:
            for i in range(len(x)):
                x[i] = prompt.format(text=x[i])
        return x, y

    def _split_generators(self, dl_manager):
        config = DATASET_CONFIG[self.config.name]
        dataset = datasets.load_dataset(
            config["name"], trust_remote_code=True
        )

        def download_custom_dataset(src_url: str, dst_path: str):
            split = src_url.split("_")[-1]
            x, y = self._prepare_dataset(dataset[split], config["input_column"], config["output_column"], config.get("prompt"))
            result_dataset = datasets.Dataset.from_dict({"input": x, "output": y})
            result_dataset.save_to_disk(dst_path)

        downloaded_files = dl_manager.download_custom(
            {split: f"{config['name']}_{split}" for split in config["splits"]}, download_custom_dataset
        )

        result = []
        if "train" in config["splits"]:
            result.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": downloaded_files["train"],
                    },
                )
            )
        if "validation" in config["splits"]:
            result.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": downloaded_files["validation"],
                    },
                )
            )
        if "test" in config["splits"]:
            result.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": downloaded_files["test"],
                    },
                )
            )

        return result

    def _generate_examples(self, filepath):
        dataset = datasets.Dataset.load_from_disk(filepath)
        for i in range(len(dataset)):
            yield i, dataset[i]
