import argparse


def parse_ssbond_records(pdb_filename):
    """
    Parses the SSBOND records from a PDB file.

    Args:
    - pdb_filename (str): The path to the PDB file.

    Returns:
    - list of dict: A list of dictionaries, each containing information about a disulfide bond.
    """
    ssbond_records = []

    with open(pdb_filename, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("SSBOND"):
                ssbond_record = {
                    "serNum": int(line[7:10].strip()),
                    "chainID1": line[15].strip(),
                    "seqNum1": int(line[17:21].strip()),
                    "icode1": line[21].strip(),
                    "chainID2": line[29].strip(),
                    "seqNum2": int(line[31:35].strip()),
                    "icode2": line[35].strip(),
                    "sym1": line[59:65].strip(),
                    "sym2": line[66:72].strip(),
                    "length": float(line[73:78].strip()),
                }
                ssbond_records.append(ssbond_record)

    return ssbond_records


def main():
    parser = argparse.ArgumentParser(
        description="Parse SSBOND records from a PDB file."
    )
    parser.add_argument("pdb_filename", type=str, help="The path to the PDB file.")
    args = parser.parse_args()

    ssbond_records = parse_ssbond_records(args.pdb_filename)
    for record in ssbond_records:
        print(record)


if __name__ == "__main__":
    main()


# end of file
