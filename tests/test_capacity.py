# tests/test_capacity.py
#
# Capacity is what stops the pipeline reporting confident states on hardware
# that cannot sample fast enough to support them. On a Raspberry Pi that is the
# difference between a monitor and a decoration, so the arithmetic is worth
# pinning down.

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from classsense import config
from classsense.capacity import (
    Capacity, default_pool_size, estimate_from_host, load, save,
)
from classsense.states import EngagementState


def make(per_face_ms=2.0, detect_ms=100.0, cores=20, pool=8,
         width=1280, every=3, gate=3):
    return Capacity(per_face_ms, detect_ms, cores, pool, width, every, gate)


class TestPoolSizing:
    """
    The hard-coded 8 was measured on a 20-core desktop and is wrong elsewhere.

    On four cores it would oversubscribe badly: each detector is already
    internally threaded, so eight of them on four cores contend rather than
    parallelise.
    """

    def test_pi_class_machine_gets_a_small_pool(self):
        assert default_pool_size(4) == 2

    def test_desktop_keeps_the_measured_knee(self):
        assert default_pool_size(20) == 8

    def test_never_drops_below_two(self):
        assert default_pool_size(1) == 2
        assert default_pool_size(2) == 2

    def test_grows_with_cores_up_to_the_knee(self):
        assert default_pool_size(8) == 4
        assert default_pool_size(12) == 6
        assert default_pool_size(64) == 8


class TestFidelityBudget:
    def test_budget_follows_the_shortest_gate(self):
        """
        Whichever temporal gate is shortest sets the sampling requirement -
        missing it is what makes a state unreliable, and the shortest is the
        easiest to miss.
        """
        cap = make(gate=3)
        shortest = min(config.SLEEPY_EYES_DURATION, config.DISTRACTED_DURATION)
        assert cap.shortest_gate == shortest
        assert cap.max_refresh_seconds == pytest.approx(shortest / 3)

    def test_stricter_sampling_shrinks_the_budget(self):
        assert make(gate=6).max_refresh_seconds < make(gate=3).max_refresh_seconds

    def test_stricter_sampling_allows_fewer_students(self):
        assert make(gate=6).compute_ceiling() < make(gate=3).compute_ceiling()


class TestComputeCeiling:
    def test_slower_faces_mean_fewer_students(self):
        assert make(per_face_ms=20.0).compute_ceiling() < make(per_face_ms=2.0).compute_ceiling()

    def test_amortising_detection_buys_students(self):
        assert make(every=6).compute_ceiling() > make(every=1).compute_ceiling()

    def test_detection_alone_exceeding_the_budget_gives_zero(self):
        """
        A real answer, and better than a number that quietly does not work.
        A 5s detection pass cannot be amortised into a 0.27s budget.
        """
        assert make(detect_ms=5000.0, every=1).compute_ceiling() == 0

    def test_a_pi_class_machine_carries_far_fewer_than_a_desktop(self):
        desktop = make(per_face_ms=2.0, detect_ms=100.0, cores=20, pool=8)
        pi = make(per_face_ms=13.0, detect_ms=320.0, cores=4, pool=2, width=640)
        assert pi.compute_ceiling() < desktop.compute_ceiling() / 3


class TestResolutionCeiling:
    """
    The second ceiling, and the one that binds first on a wide shot.

    Compute says how many faces can be processed; it says nothing about whether
    the pixels exist. Reporting compute alone would promise a cohort the camera
    cannot see, and every student past this line would be counted then
    immediately reported Unknown.
    """

    def test_more_pixels_carry_more_students(self):
        cap = make()
        assert cap.resolution_ceiling(3840, 2160) > cap.resolution_ceiling(1920, 1080)
        assert cap.resolution_ceiling(1920, 1080) > cap.resolution_ceiling(640, 480)

    def test_1080p_lands_near_the_projects_measured_design_point(self):
        """
        The pipeline was measured carrying 60 students at 1080p, so the model
        should land near that independently - otherwise it is not describing
        this system.
        """
        n = make().resolution_ceiling(1920, 1080, config.TIER_FULL_MIN_WIDTH)
        assert 45 <= n <= 90

    def test_a_low_res_source_collapses_capacity(self):
        n = make().resolution_ceiling(640, 480, config.TIER_FULL_MIN_WIDTH)
        assert n < 15

    def test_coarse_tier_permits_more_than_full(self):
        cap = make()
        assert (cap.resolution_ceiling(1920, 1080, config.TIER_COARSE_MIN_WIDTH)
                > cap.resolution_ceiling(1920, 1080, config.TIER_FULL_MIN_WIDTH))


class TestEffectiveCapacity:
    def test_capacity_is_the_lower_of_the_two_ceilings(self):
        cap = make()
        assert cap.max_students() == min(cap.compute_ceiling(),
                                         cap.resolution_ceiling())

    def test_fast_machine_is_limited_by_resolution(self):
        cap = make(per_face_ms=1.6, detect_ms=27.0, every=1)
        assert cap.limiting_factor() == "resolution"

    def test_slow_machine_is_limited_by_compute(self):
        cap = make(per_face_ms=13.0, detect_ms=320.0, cores=4, pool=2)
        assert cap.limiting_factor() == "compute"

    def test_longer_detection_interval_delays_noticing_arrivals(self):
        """Stretching detect_every buys students; admission latency is the bill."""
        assert make(every=6).admission_seconds > make(every=1).admission_seconds


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "tuning.json")
        original = make(per_face_ms=12.5, detect_ms=310.0, cores=4,
                        pool=2, width=640, every=2)
        save(original, path)

        restored = load(path)
        assert restored.source == "measured"
        assert restored.per_face_ms == pytest.approx(12.5, abs=0.01)
        assert restored.pool_size == 2
        assert restored.yolo_width == 640
        assert restored.compute_ceiling() == original.compute_ceiling()

    def test_missing_file_falls_back_to_an_estimate(self, tmp_path):
        cap = load(str(tmp_path / "does_not_exist.json"))
        assert cap.source != "measured"
        assert "ESTIMATED" in cap.source

    def test_corrupt_file_falls_back_rather_than_crashing(self, tmp_path):
        path = tmp_path / "tuning.json"
        path.write_text("{ not json at all")
        cap = load(str(path))
        assert cap.source != "measured"

    def test_incomplete_file_falls_back(self, tmp_path):
        path = tmp_path / "tuning.json"
        path.write_text(json.dumps({"cores": 4}))
        assert load(str(path)).source != "measured"

    def test_saved_dict_carries_both_ceilings(self, tmp_path):
        save(make(), str(tmp_path / "t.json"))
        data = json.loads((tmp_path / "t.json").read_text())
        for key in ("compute_ceiling", "resolution_ceiling", "max_students",
                    "limiting_factor", "admission_s"):
            assert key in data


class TestHostEstimate:
    def test_estimate_is_labelled_as_unmeasured(self):
        """
        The label is the point: an estimate that presents itself as a
        measurement is how a Pi ends up silently running desktop constants.
        """
        assert "ESTIMATED" in estimate_from_host(4).source

    def test_four_cores_estimate_far_below_twenty(self):
        assert estimate_from_host(4).compute_ceiling() < \
               estimate_from_host(20).compute_ceiling() / 3

    def test_small_hosts_get_a_narrower_detector(self):
        assert estimate_from_host(4).yolo_width < estimate_from_host(20).yolo_width


class TestUnmonitored:
    """
    What happens to students past capacity.

    Rotating everyone through at whatever rate the hardware manages produces
    states indistinguishable from trustworthy ones while being sampled too
    rarely to catch what they name - a student examined every two seconds can
    sleep through a lesson reading "Attentive".
    """

    def test_marks_state_unmonitored(self):
        s = EngagementState()
        s.mark_unmonitored()
        assert s.state == "Unmonitored"
        assert s.reason == "Beyond capacity"

    def test_claims_no_confidence(self):
        s = EngagementState()
        s.mark_unmonitored()
        assert s.confidence == 0.0

    def test_does_not_leave_a_model_query_pending(self):
        s = EngagementState()
        s.pending_model = True
        s.mark_unmonitored()
        assert s.pending_model is False

    def test_is_not_mistaken_for_attentive(self):
        s = EngagementState()
        s.mark_unmonitored()
        assert s.state not in ("Attentive", "Sleepy", "Distracted")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
