import os
import sys
import shutil
import argparse
import subprocess
import tempfile

from .resample import resample

def run(cmd, **kwargs):
    """Run a shell command, printing it and raising on failure."""
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)

def main():
    p = argparse.ArgumentParser(
        description="Batch-download, resample, and upload hyperspectral ENVI files"
    )
    p.add_argument("--pattern",  required=True,
                   help="Substring to match in remote filenames (e.g. 'YELL')")
    p.add_argument("--remote-prefix", required=True,
                   help="Remote gocmd path prefix where .hdr/.img live")
    p.add_argument("--output-prefix", required=True,
                   help="Remote gocmd path prefix to upload processed outputs")
    p.add_argument("--download-log", default="downloaded.log",
                   help="Local file to track which bases have been processed")
    args = p.parse_args()

    # load or create the download log
    done = set()
    if os.path.exists(args.download_log):
        with open(args.download_log) as f:
            done = {line.strip() for line in f}

    # 1) List remote .hdr files
    ls_cmd = ["gocmd", "ls", args.remote_prefix]
    proc = subprocess.run(ls_cmd, capture_output=True, text=True, check=True)
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip().endswith(".hdr")]

    for hdr in files:
        base = os.path.basename(hdr)
        if args.pattern not in base:
            continue
        name = base[:-4]  # strip ".hdr"
        if name in done:
            print(f"Skipping {name}, already processed")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            # 2) Download .hdr and .img
            remote_hdr = os.path.join(args.remote_prefix, base)
            remote_img = os.path.join(args.remote_prefix, name + ".img")
            local_hdr = os.path.join(tmp, os.path.basename(remote_hdr))
            local_img = os.path.join(tmp, os.path.basename(remote_img))
            run(["gocmd", "get", remote_hdr, local_hdr, "--progress"])
            run(["gocmd", "get", remote_img, local_img, "--progress"])

            # 3) Process
            output_path = resample(local_hdr)
            remote_dest = os.path.join(args.output_prefix, os.path.basename(os.path.dirname(output_path)), os.path.basename(output_path))
            run(["gocmd", "put", output_path, remote_dest])
            os.remove(output_path)

        # 5) Record completion
        with open(args.download_log, "a") as f:
            f.write(name + "\n")

if __name__ == "__main__":
    main()
