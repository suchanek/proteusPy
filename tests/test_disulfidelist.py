"""Module for testing DisulfideList class."""

# pylint: disable=C0103

import numpy as np
import pytest

from proteusPy import Disulfide, DisulfideList, load_disulfides_from_id
from proteusPy.ProteusGlobals import DATA_DIR

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0114
# pylint: disable=W0621
# pylint: disable=W0613


@pytest.fixture
def disulfide_list():
    """
    Fixture to create an empty DisulfideList for testing.

    :return: An empty DisulfideList with the name 'test'.
    :rtype: DisulfideList
    """
    sslist = DisulfideList([], "test")

    return sslist


@pytest.fixture
def pdb_5rsa():
    """
    Fixture to load disulfides from the PDB entry '5rsa'.

    :return: A DisulfideList containing disulfides from the PDB entry '5rsa', or None if no disulfides are found.
    :rtype: DisulfideList or None
    """
    entry = "5rsa"
    sslist = load_disulfides_from_id(entry, pdb_dir=DATA_DIR)

    if len(sslist) > 0:
        return sslist

    return None


def test_resolution(pdb_5rsa):
    """
    Test the resolution of the first disulfide in the PDB entry '5rsa'.

    :param pdb_5rsa: A DisulfideList containing disulfides from the PDB entry '5rsa'.
    :type pdb_5rsa: DisulfideList
    """
    assert pdb_5rsa[0].resolution == 2.0


def test_torsion_length(pdb_5rsa):
    """
    Test the torsion length of the first disulfide in the PDB entry '5rsa'.

    :param pdb_5rsa: A DisulfideList containing disulfides from the PDB entry '5rsa'.
    :type pdb_5rsa: DisulfideList
    """
    assert pdb_5rsa[0].torsion_length == pytest.approx(160.8782, rel=1e-5)


def test_calength(pdb_5rsa):
    """
    Test the CA distance of the first disulfide in the PDB entry '5rsa'.

    :param pdb_5rsa: A DisulfideList containing disulfides from the PDB entry '5rsa'.
    :type pdb_5rsa: DisulfideList
    """
    assert pdb_5rsa[0].ca_distance == pytest.approx(5.53462, rel=1e-5)


def test_average_conformation(pdb_5rsa):
    """
    Test the average conformation of the disulfides in the PDB entry '5rsa'.

    :param pdb_5rsa: A DisulfideList containing disulfides from the PDB entry '5rsa'.
    :type pdb_5rsa: DisulfideList
    """
    assert pdb_5rsa.average_conformation == pytest.approx(
        [-61.87715271, -66.73547088, -87.55935525, -72.51359397, -63.74599715],
        rel=1e-5,
    )


def test_insert_bad_entry(disulfide_list):
    """
    Test inserting a bad entry into the DisulfideList.

    :param disulfide_list: An empty DisulfideList for testing.
    :type disulfide_list: DisulfideList
    """
    item1 = "badstring"
    with pytest.raises(TypeError):
        disulfide_list.insert(0, item1)


def test_insert_at_beginning(disulfide_list):
    """
    Test inserting a Disulfide at the beginning of the DisulfideList.

    :param disulfide_list: An empty DisulfideList for testing.
    :type disulfide_list: DisulfideList
    """
    item1 = Disulfide("item1")
    disulfide_list.insert(0, item1)
    assert disulfide_list[0].name == "item1"


def test_insert_at_end(disulfide_list):
    """
    Test inserting Disulfides at the end of the DisulfideList.

    :param disulfide_list: An empty DisulfideList for testing.
    :type disulfide_list: DisulfideList
    """
    item1 = Disulfide("item1")
    item2 = Disulfide("item2")
    disulfide_list.insert(0, item1)
    disulfide_list.insert(1, item2)
    assert disulfide_list.data == [item1, item2]


def test_insert_in_middle(disulfide_list):
    """
    Test inserting a Disulfide in the middle of the DisulfideList.

    :param disulfide_list: An empty DisulfideList for testing.
    :type disulfide_list: DisulfideList
    """
    item1 = Disulfide("item1")
    item2 = Disulfide("item2")
    item3 = Disulfide("item3")

    disulfide_list.append(item1)
    disulfide_list.append(item2)
    disulfide_list.append(item3)
    assert disulfide_list.data == [item1, item2, item3]


# List Operations Tests
def test_append_invalid_type(disulfide_list):
    """Test appending an invalid type raises TypeError"""
    with pytest.raises(TypeError):
        disulfide_list.append("not a disulfide")


def test_extend_with_valid_list(disulfide_list):
    """Test extending with a valid list of disulfides"""
    items = [Disulfide("test1"), Disulfide("test2")]
    disulfide_list.extend(items)
    assert len(disulfide_list) == 2
    assert all(isinstance(item, Disulfide) for item in disulfide_list)


def test_extend_with_invalid_list(disulfide_list):
    """Test extending with invalid items raises TypeError"""
    with pytest.raises(TypeError):
        disulfide_list.extend(["not a disulfide"])


def test_getitem_with_slice(disulfide_list):
    """Test getting items with slice notation"""
    items = [Disulfide(f"test{i}") for i in range(5)]
    for item in items:
        disulfide_list.append(item)
    sliced = disulfide_list[1:4]
    assert len(sliced) == 3
    assert isinstance(sliced, DisulfideList)
    assert all(isinstance(item, Disulfide) for item in sliced)


def test_getitem_with_invalid_index(disulfide_list):
    """Test getting item with invalid index returns empty list"""
    assert len(disulfide_list[999:1000]) == 0


# Property Tests
def test_average_resolution_with_no_data(disulfide_list):
    """Test average resolution with empty list"""
    assert disulfide_list.average_resolution == -1.0


def test_average_energy_with_multiple_disulfides(disulfide_list):
    """Test average energy calculation"""
    ss1 = Disulfide("test1")
    ss1.energy = 10.0
    ss2 = Disulfide("test2")
    ss2.energy = 20.0
    disulfide_list.append(ss1)
    disulfide_list.append(ss2)
    assert disulfide_list.average_energy == 15.0


# Filter Tests
def test_filter_by_ca_distance(pdb_5rsa):
    """Test filtering by CA distance"""
    filtered = pdb_5rsa.filter_by_ca_distance(distance=6.0)
    assert isinstance(filtered, DisulfideList)
    assert all(ss.ca_distance < 6.0 for ss in filtered)


def test_filter_by_sg_distance(pdb_5rsa):
    """Test filtering by SG distance"""
    filtered = pdb_5rsa.filter_by_sg_distance(distance=3.0)
    assert isinstance(filtered, DisulfideList)
    assert all(ss.sg_distance < 3.0 for ss in filtered)


def test_filter_by_distance_with_invalid_type(pdb_5rsa):
    """Test filtering with invalid distance type raises ValueError"""
    with pytest.raises(ValueError):
        pdb_5rsa.filter_by_distance(distance_type="invalid")


# Neighbor Finding Tests
def test_nearest_neighbors_with_valid_angles(pdb_5rsa):
    """Test finding nearest neighbors with valid angles"""
    angles = [-60.0, -60.0, -90.0, -60.0, -60.0]
    neighbors = pdb_5rsa.nearest_neighbors(10.0, angles)
    assert isinstance(neighbors, DisulfideList)


def test_nearest_neighbors_with_invalid_angles(pdb_5rsa):
    """Test nearest neighbors with invalid angles raises ValueError"""
    with pytest.raises(ValueError):
        pdb_5rsa.nearest_neighbors(10.0, [-60.0])  # Not enough angles


# DataFrame Tests
def test_distance_df(pdb_5rsa):
    """Test creating distance DataFrame"""
    df = pdb_5rsa.distance_df
    assert "ca_distance" in df.columns
    assert "sg_distance" in df.columns
    assert len(df) == len(pdb_5rsa)


def test_torsion_df(pdb_5rsa):
    """Test creating torsion DataFrame"""
    df = pdb_5rsa.torsion_df
    assert len(df) == len(pdb_5rsa)


def test_torsion_array(pdb_5rsa):
    """Test getting torsion array"""
    arr = pdb_5rsa.torsion_array
    assert isinstance(arr, np.ndarray)
    assert arr.shape[1] == 5  # 5 torsion angles per disulfide


def test_minmax_energy(pdb_5rsa):
    """Test getting min/max energy disulfides"""
    min_ss, max_ss = pdb_5rsa.minmax_energy
    assert isinstance(min_ss, Disulfide)
    assert isinstance(max_ss, Disulfide)
    assert min_ss.energy <= max_ss.energy
    for ss in pdb_5rsa:
        assert min_ss.energy <= ss.energy <= max_ss.energy


def test_minmax_distance(pdb_5rsa):
    """Test getting min/max distance disulfides"""
    min_ss, max_ss = pdb_5rsa.minmax_distance()
    assert isinstance(min_ss, Disulfide)
    assert isinstance(max_ss, Disulfide)
    assert min_ss.ca_distance <= max_ss.ca_distance
    for ss in pdb_5rsa:
        assert min_ss.ca_distance <= ss.ca_distance <= max_ss.ca_distance


def test_get_by_name(pdb_5rsa):
    """Test getting disulfide by name"""
    first_ss = pdb_5rsa[0]
    found_ss = pdb_5rsa.get_by_name(first_ss.name)
    assert found_ss is not None
    assert found_ss.name == first_ss.name
    assert found_ss.energy == first_ss.energy


def test_get_by_name_nonexistent(pdb_5rsa):
    """Test getting nonexistent disulfide by name returns None"""
    found_ss = pdb_5rsa.get_by_name("nonexistent")
    assert found_ss is None


def test_center_of_mass(pdb_5rsa):
    """Test center of mass calculation"""
    com = pdb_5rsa.center_of_mass
    assert isinstance(com, np.ndarray)
    expected = np.array([-1.25358978, -1.27470883, -0.16931622])
    assert np.allclose(com, expected, rtol=1e-5)


def test_copy(pdb_5rsa):
    """Test copying a DisulfideList"""
    copied = pdb_5rsa.copy()
    assert isinstance(copied, DisulfideList)
    assert len(copied) == len(pdb_5rsa)
    assert copied is not pdb_5rsa  # Should be a new object
    assert all(a.name == b.name for a, b in zip(copied, pdb_5rsa))


def test_length_property(pdb_5rsa):
    """Test length property"""
    assert pdb_5rsa.length == len(pdb_5rsa)
    assert isinstance(pdb_5rsa.length, int)


def test_min_max_properties(pdb_5rsa):
    """Test min and max properties"""
    min_ss = pdb_5rsa.min
    max_ss = pdb_5rsa.max
    assert isinstance(min_ss, Disulfide)
    assert isinstance(max_ss, Disulfide)
    assert min_ss.energy <= max_ss.energy
    for ss in pdb_5rsa:
        assert min_ss.energy <= ss.energy <= max_ss.energy


def test_describe_method(pdb_5rsa, capsys):
    """Test describe method prints expected information"""
    pdb_5rsa.describe()
    captured = capsys.readouterr()
    output = captured.out
    assert "DisulfideList" in output
    assert "Length:" in output
    assert "Average energy:" in output
    assert "Average CA distance:" in output
    assert "Average Resolution:" in output


def test_average_torsion_distance(pdb_5rsa):
    """Test average torsion distance calculation"""
    avg_dist = pdb_5rsa.average_torsion_distance
    assert isinstance(avg_dist, float)
    assert avg_dist >= 0.0  # Distance should be non-negative


def test_build_ss_from_idlist(pdb_5rsa):
    """Test building DisulfideList from ID list"""
    # Get first disulfide's PDB ID
    first_id = pdb_5rsa[0].pdb_id
    result = pdb_5rsa.build_ss_from_idlist([first_id])
    assert isinstance(result, DisulfideList)
    assert all(ss.pdb_id == first_id for ss in result)


def test_get_chains(pdb_5rsa):
    """Test getting chain IDs"""
    chains = pdb_5rsa.get_chains()
    assert isinstance(chains, set)
    assert len(chains) > 0
    assert all(isinstance(chain, str) for chain in chains)


def test_has_chain(pdb_5rsa):
    """Test checking for chain presence"""
    # Get first chain from the list
    first_chain = next(iter(pdb_5rsa.get_chains()))
    assert pdb_5rsa.has_chain(first_chain)
    assert not pdb_5rsa.has_chain("nonexistent_chain")


def test_pprint_methods(pdb_5rsa, capsys):
    """Test pretty print methods"""
    pdb_5rsa.pprint()
    captured = capsys.readouterr()
    output1 = captured.out
    assert len(output1) > 0

    pdb_5rsa.pprint_all()
    captured = capsys.readouterr()
    output2 = captured.out
    assert len(output2) > 0
    # pprint_all should produce more output than pprint
    assert len(output2) > len(output1)


def test_resolution_property(disulfide_list):
    """Test resolution property setter"""
    new_resolution = 2.5
    disulfide_list.resolution = new_resolution
    assert disulfide_list.resolution == new_resolution
    with pytest.raises(TypeError):
        disulfide_list.resolution = "not a float"


def test_by_chain(pdb_5rsa):
    """Test filtering by chain"""
    # Get first chain
    first_chain = next(iter(pdb_5rsa.get_chains()))
    chain_list = pdb_5rsa.by_chain(first_chain)
    assert isinstance(chain_list, DisulfideList)
    assert all(ss.proximal_chain == first_chain for ss in chain_list)


def test_nearest_neighbors_ss(pdb_5rsa):
    """Test finding nearest neighbors using a reference disulfide"""
    reference_ss = pdb_5rsa[0]
    neighbors = pdb_5rsa.nearest_neighbors_ss(reference_ss, 10.0)
    assert isinstance(neighbors, DisulfideList)
    assert len(neighbors) > 0
    # Reference SS should be in its own neighborhood
    assert any(ss.name == reference_ss.name for ss in neighbors)


def test_getlist(pdb_5rsa):
    """Test getting a copy of the list"""
    copied = pdb_5rsa.getlist()
    assert isinstance(copied, DisulfideList)
    assert len(copied) == len(pdb_5rsa)
    assert copied is not pdb_5rsa  # Should be a new object
    assert all(a.name == b.name for a, b in zip(copied, pdb_5rsa))


# Comparison Operator Tests
def test_comparison_operators():
    """Test all comparison operators"""
    # Create disulfides with different energies
    ss1 = Disulfide("test1")
    ss1.energy = 10.0
    ss2 = Disulfide("test2")
    ss2.energy = 20.0

    list1 = DisulfideList([ss1], "list1")
    list2 = DisulfideList([ss2], "list2")
    list3 = DisulfideList([ss1], "list3")  # Same energy as list1

    # Test less than
    assert list1 < list2
    # Test less than or equal
    assert list1 <= list2
    assert list1 <= list3
    # Test greater than
    assert list2 > list1
    # Test greater than or equal
    assert list2 >= list1
    assert list1 >= list3
    # Test equality
    assert list1 == list3
    # Test inequality
    assert list1 != list2


# Validation Tests
def test_validate_ss_with_none(disulfide_list):
    """Test validating None raises ValueError"""
    with pytest.raises(ValueError):
        disulfide_list.validate_ss(None)


def test_validate_ss_with_invalid_type(disulfide_list):
    """Test validating invalid type raises TypeError"""
    with pytest.raises(TypeError):
        disulfide_list.validate_ss("not a disulfide")


def test_average_distance(pdb_5rsa):
    """Test average distance calculation"""
    avg_dist = pdb_5rsa.average_distance
    assert isinstance(avg_dist, float)
    assert avg_dist > 0.0  # Should be positive for real data


def test_empty_list_edge_cases(disulfide_list):
    """Test behavior with empty list"""
    assert disulfide_list.average_distance == 0.0
    assert disulfide_list.average_energy == 0.0
    assert disulfide_list.average_ca_distance == 0.0
    assert disulfide_list.minmax_distance() == (None, None)
    assert disulfide_list.minmax_energy == (None, None)
    assert len(disulfide_list.get_chains()) == 0


def test_same_chains(pdb_5rsa):
    """Test checking if disulfides are on same chains"""
    # Get first disulfide
    first_ss = pdb_5rsa[0]
    # Create a new disulfide with same chains
    same_chain_ss = Disulfide("test")
    same_chain_ss.proximal_chain = first_ss.proximal_chain
    same_chain_ss.distal_chain = first_ss.proximal_chain
    assert same_chain_ss.same_chains()


def test_setitem(disulfide_list):
    """Test setting item in list"""
    ss1 = Disulfide("test1")
    ss2 = Disulfide("test2")
    disulfide_list.append(ss1)

    # Test valid replacement
    disulfide_list[0] = ss2
    assert disulfide_list[0].name == "test2"

    # Test invalid type
    with pytest.raises(TypeError):
        disulfide_list[0] = "not a disulfide"

    # Test out of range index
    with pytest.raises(IndexError):
        disulfide_list[99] = ss1
