from setuptools import setup, find_packages
import os

install_requires = [
    'datasets>=2.3.2',
    'rouge-score==0.0.4',
    'nlpaug==1.1.10',
    'scikit-learn==1.0.2',
    'tqdm==4.64.1',
    'matplotlib==3.6',
    'pandas==1.3.5',
    'torch>=1.13.0',
    'bs4',
    'transformers>=4.30.2',
    'nltk==3.6.5',
    'sacrebleu==1.5.0',
    'sentencepiece==0.1.97',
    'hf-lfs==0.0.3',
    'pytest==4.4.1',
    'pytreebank==0.2.7',
    'setuptools==60.2.0',
    'numpy==1.23.5',
    'dill==0.3.5.1',
    'scipy==1.9.3',
    'flask==2.3.2',
    'einops',
    'accelerate',
    'bitsandbytes',
    'openai'
]

package_data = ["**/*.json", "**/*.js", "**/*.css",
                "**/*.html", "**/*.ico", "**/*.png",
                "**/assets/*", "**/assets/css/*",
                "**/assets/js/*", "**/assets/img/*"]

setup(
    name="lm_polygraph",
    version="0.0.1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="List of contributors: https://github.com/IINemo/lm-polygraph/graphs/contributors",
    author_email="artemshelmanov@gmail.com",
    description="Uncertainty Estimation Toolkit for Transformer Language Models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch uncertainty estimation",
    license="MIT",
    url="https://github.com/IINemo/lm-polygraph",
    scripts=['scripts/polygraph_eval',
             'scripts/polygraph_server'],
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"": package_data},
    python_requires=">=3.9.0",
    install_requires=list(install_requires),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
