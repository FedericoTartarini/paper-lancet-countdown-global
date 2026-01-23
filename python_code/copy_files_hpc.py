import subprocess


def sync_with_rsync():
    # Use 'sshpass' if you want to automate the password,
    # otherwise it will prompt you in the terminal.
    # On Mac: brew install sshpass

    # If you use a trailing slash on the source (daily/): It copies the contents of the folder into the destination.
    # If you do NOT use a trailing slash (daily): It copies the folder itself into the destination.
    source = "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/data/lancet/countdown-europe/data/2027/GRIDDED"
    dest = "ft8695@gadi-dm.nci.org.au:/scratch/mn51/ft8695/countdown-europe/data/2027/"

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
