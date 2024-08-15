import argparse
import os

import requests


def download_structure(pdb_id, file_format="cif", save_dir="structures"):
    """
    Download a structure from the RCSB PDB.

    :param pdb_id: The PDB ID of the structure to download.
    :param file_format: The format to download the structure in (default is 'cif').
    :param save_dir: The directory to save the downloaded structure (default is 'structures').
    """
    base_url = "https://files.rcsb.org/download"
    url = f"{base_url}/{pdb_id}.{file_format}"

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the path to save the file
    file_path = os.path.join(save_dir, f"{pdb_id}.{file_format}")

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {pdb_id}.{file_format} to {file_path}")
    else:
        print(
            f"Failed to download {pdb_id}.{file_format}. HTTP Status Code: {response.status_code}"
        )


def main():
    parser = argparse.ArgumentParser(description="Download and process PDB structures.")
    parser.add_argument(
        "pdb_id", type=str, help="The PDB ID of the structure to download."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdb",
        help="The format to download the structure in (default is 'pdb').",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="download",
        help="The directory to save the downloaded structure.",
    )

    args = parser.parse_args()
    download_structure(args.pdb_id, file_format=args.format, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
