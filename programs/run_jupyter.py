import argparse
import subprocess

import toml

# Load the pyproject.toml file
data = toml.load("pyproject.toml")

# Get the version
version = data["tool"]["poetry"]["version"]

# Create the parser
parser = argparse.ArgumentParser(
    description="Install jupyter kernel for proteusPy"
)

# Add the arguments
parser.add_argument(
    "package", type=str, help="The name of the package to install."
)

# Parse the arguments
args = parser.parse_args()

# Run the command
subprocess.run(
    [
        "python",
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        args.package,
        "--display-name",
        f"'{args.package} ({version})'",
    ]
)
