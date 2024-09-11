import argparse

from proteusPy.utility import extract_ssbonds_and_atoms


def filter_rcsb_file(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.startswith("SSBOND") or (line.startswith("ATOM") and "CYS" in line):
                outfile.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter RCSB file for SSBOND and ATOM lines containing CYS."
    )
    parser.add_argument("input_file", type=str, help="Path to the input .ent file")

    args = parser.parse_args()

    # filter_rcsb_file(args.input_file, args.output_file)

    # Usage
    ssbond_data, num_ssbonds, errors = extract_ssbonds_and_atoms(args.input_pdb_file)

    print(
        f"Number of SSBOND records in file {args.input_pdb_file}: {num_ssbonds}, Number of errors: {errors}"
    )
    # Print SSBOND records
    print("SSBOND Records:")
    for ssbond in ssbond_data["ssbonds"]:
        print(ssbond.strip())

    # Access coordinates for a specific residue

    chain_id = "A"
    res_seq_num = "25"
    for atom_name in ["N", "CA", "CB", "C", "SG"]:
        key = (chain_id, res_seq_num, atom_name)
        if key in ssbond_data["atoms"]:
            atom_record = ssbond_data["atoms"][key]
            print(
                f"Coordinates for {atom_name} in residue {res_seq_num} of chain {chain_id}: "
                f"x: {atom_record['x']}, y: {atom_record['y']}, z: {atom_record['z']}"
            )
        else:
            print(
                f"No atom data found for {atom_name} in residue {res_seq_num} of chain {chain_id}"
            )

    # end of file
