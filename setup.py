from setuptools import setup, find_packages


with open("requirements.txt") as f:
    install_requires = list(f.readlines())

package_data = [
    "**/*.json",
    "**/*.js",
    "**/*.css",
    "**/*.html",
    "**/*.ico",
    "**/*.png",
    "**/assets/*",
    "**/assets/css/*",
    "**/assets/js/*",
    "**/assets/img/*",
]

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
    scripts=[
        "scripts/polygraph_eval",
        "scripts/polygraph_server",
        "scripts/polygraph_normalize",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"": package_data},
    python_requires=">=3.10.0",
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
