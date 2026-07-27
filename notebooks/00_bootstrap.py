# First cell of every Colab notebook. Gets the repo, updates it, installs
# dependencies, and prints the commit reference for the run log.
#
# The repo is public, so no token is needed. (It goes private at submission.)

import os, sys, subprocess

REPO = "https://github.com/ookino/rlvr-argument-mining.git"
NAME = "rlvr-argument-mining"

# Make sure we are inside the repo folder.
if os.path.basename(os.getcwd()) != NAME:
    if not os.path.isdir(NAME):
        subprocess.run(["git", "clone", REPO], check=True)
    os.chdir(NAME)

# Always pull the latest code so the runtime is never stale.
subprocess.run(["git", "pull", "--quiet"], check=False)

sys.path.insert(0, os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "networkx", "pyyaml"], check=True)

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
print("commit:", commit or "(not a git checkout)")
print("ready")

# In the notebook, run these once per session for live code reloading:
#   %load_ext autoreload
#   %autoreload 2
