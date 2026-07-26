# First cell of every Colab notebook. Clones or updates the repo, installs
# dependencies, and prints the commit reference for the run log.
#
# The repo is public, so no token is needed. (It goes private at submission;
# if you ever hit a clone auth error, that switch is why.)

import os, sys, subprocess

REPO = "https://github.com/ookino/rlvr-argument-mining.git"
NAME = "rlvr-argument-mining"

if not os.path.exists("reward"):
    if not os.path.exists(NAME):
        subprocess.run(["git", "clone", REPO], check=True)
    os.chdir(NAME)
else:
    subprocess.run(["git", "pull", "--quiet"], check=False)   # already here: update

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
