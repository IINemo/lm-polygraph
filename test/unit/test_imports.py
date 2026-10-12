import subprocess
import sys


def test_numerical_modules_do_not_import_model_stack():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import lm_polygraph; import lm_polygraph.ue_metrics; "
            "import sys; "
            "assert not {'torch', 'transformers', 'vllm', 'openai'} & sys.modules.keys(); "
            "assert 'WhiteboxModel' in dir(lm_polygraph)",
        ],
        check=True,
    )
