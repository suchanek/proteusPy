import pytest

from proteusPy import Disulfide, DisulfideList, load_disulfides_from_id
from proteusPy.ProteusGlobals import DATA_DIR

# plyint: disable=C0103
# plyint: disable=C0116
# plyint: disable=C0114
# plyint: disable=W0621


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
        [
            -61.87699859,
            -66.80281181,
            -34.91612161,
            -38.36943922,
            -63.74957406,
        ],
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
