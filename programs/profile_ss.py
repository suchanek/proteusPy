import cProfile
import pstats

from proteusPy import Load_PDB_SS


def generate_test_data(length, loader):
    """
    Generate test data for DisulfideList.
    """
    # Assuming DisulfideList can be initialized with a list of random coordinates
    return loader[:length]


def profile_average_conformation(loader):
    """
    Profile the DisulfideList.average_conformation function as a function of list length.
    """
    lengths = [10, 100, 1000, 10000, 50000, 150000]  # Different lengths to test
    for length in lengths:
        profiler = cProfile.Profile()
        profiler.enable()
        test_data = generate_test_data(length, loader)
        avg = test_data.average_conformation
        profiler.disable()

        print(f"Profiling results for list length {length}:")
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(10)  # Print top 10 results
        print("\n")


if __name__ == "__main__":
    pdb = Load_PDB_SS(verbose=True, subset=False)
    profile_average_conformation(pdb)
