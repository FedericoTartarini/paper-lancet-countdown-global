from __future__ import annotations

import subprocess
from pathlib import Path

from my_config import Dirs, DirsGadi, DirsLocal

# Remote host configuration
GADI_HOST = f"{Dirs.gadi_usr}@gadi-dm.nci.org.au"


def copy_local_to_remote(
    source_dir: Path,
    remote_dest_dir: Path,
    ignore_existing: bool = True,
    file_extension: str | None = None,
):
    """
    Copy files from local machine to Gadi (remote).

    Args:
        source_dir: Local source directory path.
        remote_dest_dir: Remote destination directory path (Gadi path, without host prefix).
        ignore_existing: Whether to skip files that already exist at the destination.
        file_extension: Optional file extension filter (e.g., ".nc").
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    # Build remote destination with host prefix
    remote_dest = f"{GADI_HOST}:{remote_dest_dir}/"

    # Ensure source path ends with / to copy contents, not the directory itself
    source_str = str(source_dir).rstrip("/") + "/"

    # Use --rsync-path to create remote directory if it doesn't exist
    # Use -T flag to disable pseudo-terminal allocation (avoids .bashrc issues)
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        "ssh -T -o StrictHostKeyChecking=no -o BatchMode=yes",
        f"--rsync-path=mkdir -p {remote_dest_dir} && rsync",
    ]

    if ignore_existing:
        cmd.append("--ignore-existing")

    if file_extension:
        cmd.extend(["--include", f"*{file_extension}", "--exclude", "*"])

    cmd.extend([source_str, remote_dest])

    print("📤 Copying LOCAL → GADI")
    print(f"   From: {source_dir}")
    print(f"   To:   {remote_dest_dir}")
    print(f"   Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ rsync failed with exit code {result.returncode}")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    print("✅ Transfer complete!")


def copy_remote_to_local(
    remote_source_dir: Path,
    dest_dir: Path,
    ignore_existing: bool = True,
    file_extension: str | None = None,
    copy_subdirs: bool = True,
):
    """
    Copy files from Gadi (remote) to local machine.

    Args:
        remote_source_dir: Remote source directory path (Gadi path, without host prefix).
        dest_dir: Local destination directory path.
        ignore_existing: Whether to skip files that already exist at the destination.
        file_extension: Optional file extension filter (e.g., ".nc").
        copy_subdirs: Whether to copy subdirectories recursively.
    """
    dest_dir = Path(dest_dir)

    # Ensure local destination directory exists
    if not dest_dir.exists():
        print(f"Creating local destination directory: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Build remote source with host prefix
    remote_source = f"{GADI_HOST}:{remote_source_dir}/"

    # Use -T flag to disable pseudo-terminal allocation (avoids .bashrc issues)
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        "ssh -T -o StrictHostKeyChecking=no -o BatchMode=yes",
    ]

    if ignore_existing:
        cmd.append("--ignore-existing")

    if file_extension:
        if copy_subdirs:
            cmd.extend(
                ["--include", "*/", "--include", f"*{file_extension}", "--exclude", "*"]
            )
        else:
            cmd.extend(["--include", f"*{file_extension}", "--exclude", "*"])

    cmd.extend([remote_source, str(dest_dir) + "/"])

    print("📥 Copying GADI → LOCAL")
    print(f"   From: {remote_source_dir}")
    print(f"   To:   {dest_dir}")
    print(f"   Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ rsync failed with exit code {result.returncode}")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    print("✅ Transfer complete!")


if __name__ == "__main__":
    # Copy quantiles from local to Gadi
    copy_local_to_remote(
        source_dir=DirsLocal.e5l_q,
        remote_dest_dir=DirsGadi.e5l_q,
    )

    # Copy daily summaries from local to Gadi
    copy_local_to_remote(
        source_dir=DirsLocal.e5l_d,
        remote_dest_dir=DirsGadi.e5l_d,
    )

    # Example: Copy files from Gadi to local
    # copy_remote_to_local(
    #     remote_source_dir=DirsGadi.e5l_d,
    #     dest_dir=Path("/Users/ftar3919/Downloads/test"),
    #     file_extension=".nc",
    # )
