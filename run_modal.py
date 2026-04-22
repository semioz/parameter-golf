import modal
import subprocess
import os 
import shutil
import glob

app = modal.App("parameter-golf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "tqdm", "torch", "huggingface-hub", "setuptools",
        "typing-extensions==4.15.0", "sentencepiece", "wandb", "zstandard",
    )
)

volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu=None,
    timeout=7200,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_data():
    """Download SP4096 data on a cheap CPU instance, cache to volume."""
    os.chdir("/root")
    subprocess.run(["git", "clone", "https://github.com/semioz/parameter-golf.git"], check=True)
    os.chdir("/root/parameter-golf")

    env = os.environ.copy()
    env["MATCHED_FINEWEB_REPO_ID"] = "kevclark/parameter-golf"
    # HF_TOKEN is injected via huggingface-secret for authenticated (higher rate limit) downloads
    subprocess.run(
        ["python", "data/cached_challenge_fineweb.py", "--variant", "sp4096", "--train-shards", "80"],
        env=env, check=True,
    )

    print("Caching data to volume...")
    data_dst = "/root/parameter-golf/data"
    os.makedirs("/data/datasets", exist_ok=True)
    os.makedirs("/data/tokenizers", exist_ok=True)
    if os.path.exists(f"{data_dst}/datasets/fineweb10B_sp4096"):
        shutil.copytree(f"{data_dst}/datasets/fineweb10B_sp4096", "/data/datasets/fineweb10B_sp4096", dirs_exist_ok=True)
    for f in os.listdir(f"{data_dst}/tokenizers"):
        shutil.copy2(f"{data_dst}/tokenizers/{f}", f"/data/tokenizers/{f}")
    volume.commit()
    print("Done! Data cached to volume.")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1800,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train():
    """Run training on 8xH100. Requires data already in volume (run download_data first)."""
    os.chdir("/root")
    subprocess.run(["git", "clone", "https://github.com/semioz/parameter-golf.git"], check=True)
    os.chdir("/root/parameter-golf")

    data_dst = "/root/parameter-golf/data"
    cached_data = "/data/datasets/fineweb10B_sp4096"

    if not os.path.exists(cached_data):
        raise RuntimeError("Data not found in volume. Run `modal run run_modal.py::download_data` first.")

    print("Linking cached data from volume...")
    os.makedirs(f"{data_dst}/datasets", exist_ok=True)
    os.makedirs(f"{data_dst}/tokenizers", exist_ok=True)
    os.symlink(cached_data, f"{data_dst}/datasets/fineweb10B_sp4096")
    for f in os.listdir("/data/tokenizers"):
        src = f"/data/tokenizers/{f}"
        dst = f"{data_dst}/tokenizers/{f}"
        if not os.path.exists(dst):
            os.symlink(src, dst)

    env = os.environ.copy()
    env["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")

    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
        env=env,
    )
    print(f"Training finished with exit code {result.returncode}")

    logs = sorted(glob.glob("logs/*.txt"))
    if logs:
        with open(logs[-1]) as f:
            lines = f.readlines()
        for line in lines[-30:]:
            print(line, end="")


@app.local_entrypoint()
def main():
    train.remote()
