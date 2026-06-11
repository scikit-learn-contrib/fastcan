import os
import subprocess

REPO_URL = "{REPO_URL}"
BRANCH = "{BRANCH}"


def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    run_cmd(f"git clone --branch {BRANCH} {REPO_URL} fastcan_repo")
    os.chdir("fastcan_repo")
    run_cmd("curl -fsSL https://pixi.sh/install.sh | bash")
    home = os.path.expanduser("~")
    pixi_bin = os.path.join(home, ".pixi", "bin")
    os.environ["PATH"] = f"{pixi_bin}:{os.environ.get('PATH', '')}"
    run_cmd("pixi run -e dev test-gpu")


if __name__ == "__main__":
    main()
