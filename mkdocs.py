# mymodule.py

import subprocess


def main():
    subprocess.run(
        ["pdoc", "-o", "docs", "--math", "--logo", "./logo.png", "./proteusPy"]
    )
