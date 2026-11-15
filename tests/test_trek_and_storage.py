"""Moon Trek tile addressing and the operation store (no network, no NASA data needed)."""
import numpy as np

from api.storage import OperationStore
from data_engine.trek_client import resolution_m, tile_for_latlon


def test_trek_tile_matches_live_verified_tile():
    t = tile_for_latlon("lola_color_shade", 9, -89.625, 119.13)  # Shackleton floor
    assert (t["row"], t["col"]) == (257, 258)  # fetched live: HTTP 200 image/png
    assert t["url"].endswith("/9/257/258.png")
    assert abs(resolution_m("lola_color_shade", 9) - 16.72) < 0.01


def test_operation_store_crud_and_numpy_values(tmp_path):
    s = OperationStore(tmp_path / "ops.sqlite3")
    a = s.create("first", "analysis", {"lat": -85.0}, {"n": np.int64(3), "ok": np.bool_(True), "x": np.float64(0.5)})
    b = s.create("second", "comparison", {}, {})
    assert [o["id"] for o in s.list()] == [b["id"], a["id"]]  # newest first
    assert [o["id"] for o in s.list(kind="analysis")] == [a["id"]]
    assert s.get(a["id"])["result"] == {"n": 3, "ok": True, "x": 0.5}
    assert s.delete(a["id"]) and not s.delete(a["id"]) and s.get(a["id"]) is None
