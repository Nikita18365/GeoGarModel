import pytest
from geogarmodel.parameter_files import best_snapshot, read_parameters


def test_best_snapshot_merges_fixed_and_ignores_incomplete():
    context = {"model": "Ridge", "parameters": {"alpha": 3, "fit_intercept": False}}
    history = [dict(trial=0, state="COMPLETE", rmse=2., parameters={"alpha": .5}),
               dict(trial=1, state="PRUNED", rmse=.1, parameters={"alpha": .01}),
               dict(trial=2, state="FAIL", rmse=None, parameters={})]
    snapshot = best_snapshot(context, history)
    assert snapshot["best_trial"] == 0
    assert read_parameters(snapshot) == ("Ridge", {"alpha": .5, "fit_intercept": False})
    history.append(dict(trial=3, state="COMPLETE", rmse=1., parameters={"alpha": 1.}))
    assert snapshot["best_trial"] == 0
    assert best_snapshot(context, history)["best_trial"] == 3


def test_import_report_plain_and_no_complete():
    assert read_parameters({"alpha": 2}) == (None, {"alpha": 2})
    assert read_parameters({"model": "Ridge", "settings": {
        "parameters": {"fit_intercept": False}, "best_parameters": {"alpha": 2}}}) == (
            "Ridge", {"fit_intercept": False, "alpha": 2})
    with pytest.raises(ValueError, match="COMPLETE"):
        best_snapshot({}, [])
    with pytest.raises(ValueError):
        read_parameters({"device": "cuda"})
