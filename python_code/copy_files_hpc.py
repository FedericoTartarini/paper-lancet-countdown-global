import subprocess
from pathlib import Path

from my_config import DirsGadi


def sync_with_rsync(source: str = None, dest: Path = None):
    # Use 'sshpass' if you want to automate the password,
    # otherwise it will prompt you in the terminal.
    # On Mac: brew install sshpass

    # If you use a trailing slash on the source (daily/): It copies the contents of the folder into the destination.
    # If you do NOT use a trailing slash (daily): It copies the folder itself into the destination.

    # Command explanation:
    # -a: Archive mode (keeps permissions/dates)
    # -v: Verbose#
    # --progress: Shows the progress bar
    # --ignore-existing: Skips files that exist on Gadi

    cmd = [
        "rsync",
        "-avP",
        "--progress",
        "--ignore-existing",
        "-e",
        "ssh -o StrictHostKeyChecking=no",
        source,
        dest,
    ]

    print("Starting Rsync...")
    subprocess.run(cmd)


def copy_files_from_remote(
    source_dir: str,
    dest_dir: Path,
    copy_subdirs: bool = True,
    file_extension: str = None,
) -> None:
    """
    Copies files from a remote source directory to a local destination directory.
    Does not overwrite existing local files. Optionally filters by file extension and controls subdirectory copying.

    Args:
        source_dir (str): Remote source directory path (e.g., 'user@host:/path/to/dir').
        dest_dir (Path): Local destination directory path.
        copy_subdirs (bool): If True, copies subdirectories recursively. Defaults to True.
        file_extension (str, optional): File extension to filter (e.g., '.nc'). If None, copies all files.
    """
    cmd = [
        "rsync",
        "-avP",
        "--progress",
        "--ignore-existing",
        "-e",
        "ssh -o StrictHostKeyChecking=no",
    ]
    if not copy_subdirs:
        cmd.append("--no-recursive")
    if file_extension:
        cmd.extend(["--include", f"*{file_extension}", "--exclude", "*"])
    cmd.extend([source_dir, str(dest_dir)])
    print(f"Starting Rsync from {source_dir} to {dest_dir}...")
    subprocess.run(cmd)


if __name__ == "__main__":
    # Example usage: Copy ERA5-Land hourly data for 2020 from Gadi to local
    copy_files_from_remote(
        source_dir=f"ft8695@gadi-dm.nci.org.au:{DirsGadi.e5l_d}",
        dest_dir=Path("/Users/ftar3919/Downloads/test"),
        copy_subdirs=True,
        file_extension=None,  # Temporarily remove filter to debug
    )
