# Paste this as the FIRST CELL of every Colab notebook.
# It clones the repo, installs, and turns on live code reloading so you can
# edit a .py file and re-run a cell without restarting or reloading models.

import os, subprocess, sys

REPO = "https://github.com/YOURNAME/rlvr-argument-mining.git"   # <-- set this
NAME = REPO.rstrip("/").split("/")[-1].replace(".git", "")

if not os.path.exists(NAME):
    subprocess.run(["git", "clone", REPO], check=True)
os.chdir(NAME)
subprocess.run(["git", "pull"], check=False)
sys.path.insert(0, os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

# Record which version of the code produced this run. Paste the output into
# docs/run_log.md alongside the result. This is most of your reproducibility marks.
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
print("running commit:", commit)

# In the notebook itself, run these as a separate cell:
#   %load_ext autoreload
#   %autoreload 2
