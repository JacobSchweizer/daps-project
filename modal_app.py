# modal_app.py
from modal import Image, App, Volume
import os, subprocess, sys, pathlib, shutil, re

# ------------------ Image definition ------------------

image = (
    Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    .entrypoint([])  # don’t spam logs
    .apt_install("git", "wget", "unzip")
    .pip_install("pip", "setuptools", "wheel", "cython<3")
    # Preinstall Torch 2.3.1 cu121 stack
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"PIP_DEFAULT_TIMEOUT": "1000", "PIP_RETRIES": "5"})
    # Bake your repo snapshot LAST so requirements.txt and code are included
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[
            ".git", ".venv", "venv", "__pycache__",
            ".mypy_cache", ".pytest_cache", ".DS_Store",
            "results",
        ],
    )
)

# ------------------ Persistent volume ------------------

vol = Volume.from_name("daps-data", create_if_missing=True)
app = App("daps-gpu")


# ------------------ Helpers ------------------

def _ensure_symlink_to_vol(name: str):
    """Make ./name a symlink to /vol/name (persistent across runs)."""
    os.makedirs(f"/vol/{name}", exist_ok=True)
    p = pathlib.Path(name)

    if p.is_symlink():
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    elif p.exists() and p.is_dir() and not any(p.iterdir()):
        # empty real dir -> remove so we can replace with symlink
        shutil.rmtree(p)

    if not p.exists():
        os.symlink(f"/vol/{name}", name)


def _seed_from_workspace(subdir: str):
    """
    If /vol/<subdir> is empty but /workspace/<subdir> exists (baked from local),
    copy its contents into the volume once.
    """
    src = pathlib.Path("/workspace") / subdir
    dst = pathlib.Path("/vol") / subdir

    if src.exists():
        empty = (not dst.exists()) or (dst.is_dir() and not any(dst.iterdir()))
        if empty:
            print(f"📦 Seeding /vol/{subdir} from /workspace/{subdir} ...")
            os.makedirs(dst, exist_ok=True)
            subprocess.check_call(["bash", "-lc", f"cp -a {src}/. {dst}/"])


# ------------------ Main experiment ------------------

@app.function(image=image, gpu="A100-80GB", timeout=2 * 60 * 60, volumes={"/vol": vol})
def run_experiment(
    reqs_path: str = "/workspace/requirements.txt",
    data: str = "test-ffhq",
    model: str = "ffhq256ddpm",
    save_dir: str = "results/pixel/ffhq",
    batch_size: int = 32,
    num_runs: int = 4,
    name: str = "phase_retrieval",
    task: str = "phase_retrieval",   # 👈 NEW: task parameter
):
    os.chdir("/workspace")

    # 1) Install Python deps, but skip torch/vision/audio (already preinstalled)
    wheel_dir = "/vol/pip-wheels"
    os.makedirs(wheel_dir, exist_ok=True)

    if os.path.exists(reqs_path):
        print(f"📦 Installing project requirements from {reqs_path}...")
        filtered_reqs = "/tmp/requirements.filtered.txt"

        skip_re = re.compile(
            r"^\s*(torch|torchvision|torchaudio|triton|pytorch-cuda)\b",
            re.IGNORECASE,
        )

        with open(reqs_path, "r") as f_in, open(filtered_reqs, "w") as f_out:
            for line in f_in:
                if skip_re.match(line):
                    print(f"↪️  Skipping pinned torch line: {line.strip()}")
                    continue
                f_out.write(line)

        subprocess.call([sys.executable, "-m", "pip", "install", "-U", "pip"])

        # Try wheel cache first
        try:
            subprocess.call([
                sys.executable, "-m", "pip", "download",
                "-r", filtered_reqs, "-d", wheel_dir,
                "--retries", "5", "--timeout", "1000",
                "-i", "https://pypi.org/simple",
            ])
        except Exception:
            print("⚠️  pip download hiccup; continuing.")

        try:
            subprocess.call([
                sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", wheel_dir,
                "-r", filtered_reqs,
            ])
        except Exception:
            subprocess.call([
                sys.executable, "-m", "pip", "install",
                "-r", filtered_reqs,
                "--retries", "5", "--timeout", "1000",
                "-i", "https://pypi.org/simple",
            ])
    else:
        print(f"ℹ️ No requirements file at {reqs_path}; continuing without extra deps.")

    # 2) Symlink persistent dirs and seed from your baked repo
    for d in ["checkpoints", "dataset", "results"]:
        _ensure_symlink_to_vol(d)

    _seed_from_workspace("checkpoints")
    _seed_from_workspace("dataset")

    # 3) Sanity checks for checkpoints
    need = [
        "checkpoints/ffhq256.pt",        # pixel DDPM (FFHQ)
        "checkpoints/imagenet256.pt",    # pixel DDPM (ImageNet)
        "checkpoints/ldm_ffhq256.pt",    # LDM
        "checkpoints/ldm_imagenet256.pt" # LDM
    ]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("❌ Missing required files:\n  " + "\n  ".join(missing))
        sys.exit(1)

    # 4) Dataset path
    ds_path = pathlib.Path("dataset") / data
    if not ds_path.exists():
        print(f"❌ Dataset not found at: {ds_path}")
        sys.exit(1)
    else:
        print(f"✅ Dataset found at: {ds_path}")
        files = list(ds_path.glob("*"))
        print(f"📂 Found {len(files)} files (showing first 5):")
        for f in files[:5]:
            print("   ", f)

    # 5) Ensure save dir exists
    os.makedirs(save_dir, exist_ok=True)

    # 6) Build and run the posterior sampling command
    cmd = [
        "python", "posterior_sample.py",
        f"+data={data}",
        f"+model={model}",
        f"+task={task}",                     # 👈 now configurable
        "+sampler=edm_daps",
        "task_group=pixel",
        f"save_dir={save_dir}",
        f"num_runs={num_runs}",
        "sampler.diffusion_scheduler_config.num_steps=5",
        "sampler.annealing_scheduler_config.num_steps=200",
        f"batch_size={batch_size}",
        f"name={name}",
        "gpu=0",                            # single GPU in Modal container
    ]

    print("🚀 Running:", " ".join(cmd))
    rc = subprocess.call(cmd)

    vol.commit()
    sys.exit(rc)


# ------------------ Local entrypoint ------------------

@app.local_entrypoint()
def main(
    reqs_path: str = "/workspace/requirements.txt",
    data: str = "test-ffhq",
    model: str = "ffhq256ddpm",
    save_dir: str = "results/pixel/ffhq",
    batch_size: int = 32,
    num_runs: int = 4,
    name: str = "phase_retrieval",
    task: str = "phase_retrieval",
):
    """
    Called when you do:
      modal run modal_app.py --data ... --model ... --task ...
    """
    run_experiment.remote(
        reqs_path=reqs_path,
        data=data,
        model=model,
        save_dir=save_dir,
        batch_size=batch_size,
        num_runs=num_runs,
        name=name,
        task=task,
    )
