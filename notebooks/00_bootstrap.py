# First cell of every Colab notebook. Clones or updates the repo, installs
# dependencies, and prints the commit reference for the run log.
#
# The repo is PRIVATE, so cloning it on Colab needs a read-only access token.
# Store the token once in Colab Secrets (the key icon in the sidebar) under the
# name GITHUB_TOKEN, with Notebook access turned on. The token is read from
# there and never written into the notebook or the stored git remote.

import os, sys, subprocess

REPO_USER = "ookino"
REPO_NAME = "rlvr-argument-mining"
CLEAN_URL = f"https://github.com/{REPO_USER}/{REPO_NAME}.git"


def _token():
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None   # not on Colab, or no secret set


def _clone():
    token = _token()
    url = f"https://{token}@github.com/{REPO_USER}/{REPO_NAME}.git" if token else CLEAN_URL
    try:
        subprocess.run(["git", "clone", url, REPO_NAME], check=True)
    except subprocess.CalledProcessError:
        raise SystemExit(
            "Could not clone the private repo. Add a read-only GITHUB_TOKEN in "
            "Colab Secrets (key icon, left sidebar) and re-run. See notebook "
            "instructions."
        )
    # Never leave the token sitting in the stored remote URL.
    subprocess.run(["git", "-C", REPO_NAME, "remote", "set-url", "origin", CLEAN_URL], check=True)
    os.chdir(REPO_NAME)


if not os.path.exists("reward"):
    _clone()

sys.path.insert(0, os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "networkx", "pyyaml"], check=True)

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
print("commit:", commit or "(not a git checkout)")
print("ready")

# In the notebook itself, run these as a separate cell for live code reloading:
#   %load_ext autoreload
#   %autoreload 2
