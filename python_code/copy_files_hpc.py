import subprocess
from pathlib import Path


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


if __name__ == "__main__":
    sync_with_rsync()
