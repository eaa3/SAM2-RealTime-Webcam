import subprocess
import sys
import requests

def check_command(command):
    """Check if a command is available on the system."""
    try:
        subprocess.run([command, '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_file(url, filename):
    """Download a file from the given URL."""
    try:
        print(f"Downloading {filename} checkpoint...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully.")
    except Exception as e:
        print(f"Failed to download checkpoint from {url}: {e}")
        sys.exit(1)

def download_checkpoints(base_url):
    """Download multiple checkpoints from the given base URL."""
    print("Checking for wget or curl...")
    if check_command("wget"):
        print("wget is available.")
    elif check_command("curl"):
        print("curl is available.")
    else:
        print("Please install wget or curl to download the checkpoints.")
        sys.exit(1)

    urls_and_files = [
        (f"{base_url}/sam2.1_hiera_tiny.pt", "sam2.1_hiera_tiny.pt"),
        (f"{base_url}/sam2.1_hiera_small.pt", "sam2.1_hiera_small.pt"),
        (f"{base_url}/sam2.1_hiera_base_plus.pt", "sam2.1_hiera_base_plus.pt"),
        (f"{base_url}/sam2.1_hiera_large.pt", "sam2.1_hiera_large.pt"),
    ]

    # Download each checkpoint
    for url, filename in urls_and_files:
        download_file(url, filename)

    print("All checkpoints are downloaded successfully.")

if __name__ == "__main__":
    SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    download_checkpoints(SAM2p1_BASE_URL)