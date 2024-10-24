import subprocess


def bash(cmd):
    return subprocess.run(cmd, shell=True)

for dataset in "wmt14_fren,wmt19_deen".split(","):
    print("Processing dataset", dataset)
    bash(f"python3 manager.py --dataset {dataset} -p")
