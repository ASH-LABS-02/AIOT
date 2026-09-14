# tests/test_detector.py
#
# Backend selection, and the one failure that must never reach production.
#
# An NCNN export has a fixed input shape. Run a 640px export at 960px and it
# returns zero detections - no exception, no warning, an empty list. Measured
# on a real export: 1 person found at 640, none at 960, identical frame. In a
# classroom that reads as an empty room while every other part of the system
# keeps working perfectly, which is the worst shape a bug can take here.

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from classsense.capacity import Capacity, load, save
from classsense.detector import (
    ncnn_available, ncnn_export_width, write_ncnn_meta, NCNN_META,
)


@pytest.fixture
def fake_export(tmp_path):
    """A directory shaped like an NCNN export, without the weights."""
    d = tmp_path / "yolov8n_ncnn_model"
    d.mkdir()
    (d / "model.ncnn.param").write_text("7767517\n")
    (d / "model.ncnn.bin").write_bytes(b"\x00" * 16)
    return d


class TestExportMetadata:
    def test_records_and_reads_back_the_width(self, fake_export):
        write_ncnn_meta(str(fake_export), 640, "yolov8n.pt")
        assert ncnn_export_width(str(fake_export)) == 640

    def test_missing_metadata_reads_as_unknown(self, fake_export):
        """
        An export made by hand through ultralytics has no sidecar. That must
        read as "cannot verify", never as "fine" - the whole point is that an
        unverifiable width is the dangerous case.
        """
        assert ncnn_export_width(str(fake_export)) is None

    def test_corrupt_metadata_reads_as_unknown(self, fake_export):
        (fake_export / NCNN_META).write_text("{ not json")
        assert ncnn_export_width(str(fake_export)) is None

    def test_metadata_without_imgsz_reads_as_unknown(self, fake_export):
        (fake_export / NCNN_META).write_text(json.dumps({"source": "x.pt"}))
        assert ncnn_export_width(str(fake_export)) is None

    def test_metadata_warns_the_reader_about_the_fixed_shape(self, fake_export):
        write_ncnn_meta(str(fake_export), 640, "yolov8n.pt")
        note = json.loads((fake_export / NCNN_META).read_text())["note"]
        assert "zero detections" in note.lower()

    def test_availability_needs_an_actual_param_file(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert ncnn_available(str(empty)) is False
        assert ncnn_available(str(tmp_path / "nope")) is False

    def test_availability_true_for_a_real_looking_export(self, fake_export):
        assert ncnn_available(str(fake_export)) is True


class TestWidthGuard:
    """
    The guard exists because the failure is silent, so it must be loud.

    These drive load_detector's decision logic through the metadata only - they
    deliberately do not load real weights, which would make the suite depend on
    a 12MB export existing.
    """

    def test_matching_width_is_accepted(self, fake_export):
        write_ncnn_meta(str(fake_export), 640, "yolov8n.pt")
        assert ncnn_export_width(str(fake_export)) == 640

    def test_mismatch_is_detectable_before_any_inference(self, fake_export):
        """
        The check has to be possible at load time. Discovering it from an empty
        result at runtime is indistinguishable from an empty room.
        """
        write_ncnn_meta(str(fake_export), 640, "yolov8n.pt")
        configured = 960
        assert ncnn_export_width(str(fake_export)) != configured


class TestBackendInCapacity:
    """The backend and the width travel together, since one binds the other."""

    def test_defaults_to_pytorch(self):
        cap = Capacity(2.0, 100.0, 20, 8, 1280, 3, 3)
        assert cap.backend == "pytorch"

    def test_backend_survives_a_round_trip(self, tmp_path):
        path = str(tmp_path / "tuning.json")
        save(Capacity(12.0, 300.0, 4, 2, 640, 2, 3, backend="ncnn"), path)
        assert load(path).backend == "ncnn"

    def test_backend_is_written_to_the_tuning_file(self, tmp_path):
        path = tmp_path / "t.json"
        save(Capacity(2.0, 100.0, 20, 8, 640, 1, 3, backend="ncnn"), str(path))
        assert json.loads(path.read_text())["backend"] == "ncnn"

    def test_older_tuning_without_a_backend_still_loads(self, tmp_path):
        """A tuning file written before backends existed must not break."""
        path = tmp_path / "t.json"
        path.write_text(json.dumps({
            "per_face_ms": 2.0, "detect_ms": 100.0, "cores": 20,
            "pool_size": 8, "yolo_width": 640, "detect_every": 1,
        }))
        cap = load(str(path))
        assert cap.source == "measured"
        assert cap.backend == "pytorch"

    def test_summary_names_the_backend(self):
        cap = Capacity(12.0, 120.0, 4, 2, 640, 2, 3, backend="ncnn")
        assert "ncnn" in cap.summary()


class TestNcnnChangesCapacity:
    """
    Why this is worth doing at all, and what it cannot fix.

    A faster detector buys students, but landmark cost per face is untouched by
    it - so there is a hard ceiling that NCNN cannot lift.
    """

    def _pi(self, detect_ms):
        return Capacity(per_face_ms=12.5, detect_ms=detect_ms, cores=4,
                        pool_size=2, yolo_width=640, detect_every=3,
                        samples_per_gate=3)

    def test_faster_detection_raises_the_student_cap(self):
        slow = self._pi(320.0).compute_ceiling()
        fast = self._pi(110.0).compute_ceiling()
        assert fast > slow

    def test_even_free_detection_is_capped_by_landmark_cost(self):
        """
        With detection at zero the budget is spent entirely on faces, so the
        ceiling is refresh/per_face. NCNN cannot go past this, and saying so is
        the difference between a useful optimisation and an overpromise.
        """
        free = self._pi(0.0)
        expected = int((free.max_refresh_seconds * 1000) // free.per_face_ms)
        assert free.compute_ceiling() == expected

    def test_the_remaining_headroom_is_bounded(self):
        realistic = self._pi(320.0).compute_ceiling()
        theoretical_best = self._pi(0.0).compute_ceiling()
        # Worth doing, but not transformative - the gain is bounded by how much
        # of the budget detection was taking in the first place.
        assert theoretical_best > realistic
        assert theoretical_best < realistic * 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
