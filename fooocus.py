import os
import sys
import shutil
import subprocess
import signal
from urllib.request import Request, urlopen
from tqdm import tqdm  # Progress bar library

# Constants
MODEL_URL = "https://huggingface.co/oieieio/juggernautXL_v8Rundiffusion/resolve/main/juggernautXL_v8Rundiffusion.safetensors"
MODEL_PATH = "models/checkpoints/juggernautXL_v8Rundiffusion.safetensors"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Cleanup temporary directory
def cleanup_temp_dir():
    temp_dir = "/tmp/fooocus"
    print(f"[Cleanup] Attempting to delete content of temp dir {temp_dir}")
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("[Cleanup] Cleanup successful")
    except Exception as e:
        print(f"[Cleanup] Failed to delete content of temp dir: {e}")

# Download the model with progress
def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"[Model Download] Model already exists at {target_path}")
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"[Model Download] Downloading: \"{url}\" to {target_path}")

    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"} if HUGGINGFACE_TOKEN else {}
    req = Request(url, headers=headers)

    try:
        with urlopen(req) as response, open(target_path, "wb") as out_file:
            total_size = int(response.getheader("Content-Length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as progress_bar:
                for chunk in iter(lambda: response.read(1024 * 8), b""):
                    out_file.write(chunk)
                    progress_bar.update(len(chunk))
        print(f"[Model Download] Successfully downloaded model to {target_path}")
    except Exception as e:
        print(f"[Model Download] Failed to download {url}. Error: {e}")
        sys.exit(1)

# Launch Fooocus application
def launch_fooocus():
    print("[Fooocus] Launching application...")
    try:
        subprocess.run([sys.executable, "launch.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Fooocus] Failed to launch application: {e}")
        sys.exit(1)

# Signal handling for graceful shutdown
def signal_handler(signum, frame):
    print("[Fooocus] Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main logic
def main():
    print(f"Python {sys.version}")
    print("Fooocus version: 2.5.5")

    cleanup_temp_dir()
    download_file(MODEL_URL, MODEL_PATH)
    launch_fooocus()

if __name__ == "__main__":
    main()
