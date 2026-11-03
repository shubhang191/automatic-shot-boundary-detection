import pytest
from sbd_app.shot_schema import CandidateCut as Cut, canonical_cuts, shots_from_cuts, cuts_from_native_scenes


@pytest.mark.parametrize("cuts,count", [([], 1), ([], 15), ([Cut(4,5)],10), ([Cut(0,1),Cut(1,2)],3), ([Cut(3,7)],12), ([Cut(3,7),Cut(3,7)],12)])
def test_partition(cuts, count):
    shots = shots_from_cuts(cuts, count, 30000 / 1001)
    assert [i for s in shots for i in range(s["start_frame"],s["end_frame"]+1)] == list(range(count))
    assert abs(sum(s["duration_s"] for s in shots) - count * 1001 / 30000) < 1e-12
    for s in shots:
        assert round(s["start_s"] * 30000 / 1001) == s["start_frame"]
        assert round(s["end_exclusive_s"] * 30000 / 1001) - 1 == s["end_frame"]


@pytest.mark.parametrize("cuts,count", [([],0),([Cut(-1,1)],3),([Cut(0,3)],3),([Cut(1,1)],3),([Cut(0.,1)],3),([Cut(0,2),Cut(1,3)],5)])
def test_invalid(cuts,count):
    with pytest.raises(ValueError):canonical_cuts(cuts,count)


def test_gradual_allocation():
    shots=shots_from_cuts([Cut(3,7)],10,25)
    assert [(s["start_frame"],s["end_frame"]) for s in shots]==[(0,6),(7,9)]
    assert cuts_from_native_scenes([[0,3],[7,9]],10)==[Cut(3,7)]
