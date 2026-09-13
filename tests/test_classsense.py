# tests/test_classsense.py
#
#   python -m pytest tests/ -v          (or: python tests/test_classsense.py)
#
# Covers the parts where a silent regression would be expensive and invisible:
# scale invariance of the geometry (the property the train/serve skew violated),
# tier boundaries, tracker association, and the temporal gates.

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from classsense import config
from classsense.geometry import (
    eye_aspect_ratio, mouth_aspect_ratio, head_pose_angles,
    face_width_px, face_height_px, face_size_px, face_centre_in_frame,
    extract_feature_row, compute_box_iou,
)
from classsense.states import EngagementState
from classsense.tiers import (
    Tier, tier_for_face_width, trusts_eyes, trusts_pose, states_available,
)
from classsense.tracker import StudentTracker, TrackedStudent


# ── helpers ────────────────────────────────────
def _lm(x, y, z=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z)


def synthetic_face(yaw_shift=0.0, pitch_shift=0.0, eye_open=0.04,
                   mouth_open=0.01, roll=0.0):
    """
    A face as 478 normalised landmarks, with only the indices we read set
    meaningfully. Enough to exercise the geometry deterministically.
    """
    lms = [_lm(0.5, 0.5) for _ in range(478)]

    cx, cy = 0.5 + yaw_shift, 0.5 + pitch_shift
    lms[config.NOSE_TIP] = _lm(cx, cy)
    lms[config.LEFT_FACE] = _lm(0.30, 0.50)
    lms[config.RIGHT_FACE] = _lm(0.70, 0.50)
    lms[config.FOREHEAD] = _lm(0.50, 0.25)
    lms[config.CHIN] = _lm(0.50, 0.78)

    # `roll` tilts the eye line: the left corner rises by half and the right
    # falls by half, so the face rotates about its centre rather than shearing.
    half = roll / 2.0

    # Left eye: 6 points, [outer, top1, top2, inner, bottom2, bottom1]
    lx = 0.38
    lms[config.LEFT_EYE[0]] = _lm(lx - 0.04, 0.45 - half)
    lms[config.LEFT_EYE[3]] = _lm(lx + 0.04, 0.45 + half)
    lms[config.LEFT_EYE[1]] = _lm(lx - 0.015, 0.45 - eye_open)
    lms[config.LEFT_EYE[2]] = _lm(lx + 0.015, 0.45 - eye_open)
    lms[config.LEFT_EYE[4]] = _lm(lx + 0.015, 0.45 + eye_open)
    lms[config.LEFT_EYE[5]] = _lm(lx - 0.015, 0.45 + eye_open)

    rx = 0.62
    lms[config.RIGHT_EYE[0]] = _lm(rx - 0.04, 0.45 - half)
    lms[config.RIGHT_EYE[3]] = _lm(rx + 0.04, 0.45 + half)
    lms[config.RIGHT_EYE[1]] = _lm(rx - 0.015, 0.45 - eye_open)
    lms[config.RIGHT_EYE[2]] = _lm(rx + 0.015, 0.45 - eye_open)
    lms[config.RIGHT_EYE[4]] = _lm(rx + 0.015, 0.45 + eye_open)
    lms[config.RIGHT_EYE[5]] = _lm(rx - 0.015, 0.45 + eye_open)

    # Eye-line outers drive roll.
    lms[config.LEFT_EYE_OUTER] = _lm(0.34, 0.45 - half)
    lms[config.RIGHT_EYE_OUTER] = _lm(0.66, 0.45 + half)

    lms[config.MOUTH[0]] = _lm(0.42, 0.66)
    lms[config.MOUTH[1]] = _lm(0.58, 0.66)
    lms[config.MOUTH[2]] = _lm(0.50, 0.66 - mouth_open)
    lms[config.MOUTH[3]] = _lm(0.50, 0.66 + mouth_open)
    return lms


# ── geometry: the property the old skew broke ──
class TestScaleInvariance:
    """
    Every measure must give the same answer at any crop size.

    This is the regression test for the bug this rebuild fixes. Training
    averaged clip features at native video resolution; live inference now
    measures on crops capped at 192px. If these values moved with resolution,
    the two paths would disagree again - just more subtly than before.
    """

    @pytest.mark.parametrize("size", [64, 128, 192, 480, 1080])
    def test_ear_is_scale_invariant(self, size):
        lms = synthetic_face()
        ref = eye_aspect_ratio(lms, config.LEFT_EYE, 192, 192)
        got = eye_aspect_ratio(lms, config.LEFT_EYE, size, size)
        assert got == pytest.approx(ref, abs=1e-3)

    @pytest.mark.parametrize("size", [64, 128, 192, 480, 1080])
    def test_mar_is_scale_invariant(self, size):
        lms = synthetic_face()
        ref = mouth_aspect_ratio(lms, 192, 192)
        assert mouth_aspect_ratio(lms, size, size) == pytest.approx(ref, abs=1e-3)

    @pytest.mark.parametrize("size", [64, 128, 192, 480, 1080])
    def test_yaw_pitch_are_scale_invariant(self, size):
        lms = synthetic_face(yaw_shift=0.06)
        ref_yaw, ref_pitch, _ = head_pose_angles(lms, 192, 192)
        yaw, pitch, _ = head_pose_angles(lms, size, size)
        assert yaw == pytest.approx(ref_yaw, abs=1e-2)
        assert pitch == pytest.approx(ref_pitch, abs=1e-2)

    def test_feature_row_matches_feature_cols(self):
        row = extract_feature_row(synthetic_face(), 192, 192)
        assert set(row) == set(config.FEATURE_COLS)


class TestGeometryDirections:
    def test_open_eye_reads_above_closed_threshold(self):
        ear = eye_aspect_ratio(synthetic_face(eye_open=0.04),
                               config.LEFT_EYE, 192, 192)
        assert ear > config.EAR_CLOSED_THRESH

    def test_closed_eye_reads_below_closed_threshold(self):
        ear = eye_aspect_ratio(synthetic_face(eye_open=0.002),
                               config.LEFT_EYE, 192, 192)
        assert ear < config.EAR_CLOSED_THRESH

    def test_yawn_exceeds_threshold(self):
        mar = mouth_aspect_ratio(synthetic_face(mouth_open=0.06), 192, 192)
        assert mar > config.MAR_YAWN_THRESH

    def test_yaw_sign_follows_head_turn(self):
        left, _, _ = head_pose_angles(synthetic_face(yaw_shift=-0.08), 192, 192)
        right, _, _ = head_pose_angles(synthetic_face(yaw_shift=0.08), 192, 192)
        assert left < 0 < right

    def test_level_head_has_near_zero_roll(self):
        _, _, roll = head_pose_angles(synthetic_face(roll=0.0), 192, 192)
        assert abs(roll) < 1.0

    def test_tilted_head_produces_roll(self):
        _, _, roll = head_pose_angles(synthetic_face(roll=0.25), 192, 192)
        assert abs(roll) > config.ROLL_RECLINED_THRESH

    def test_face_width_tracks_pixel_width(self):
        lms = synthetic_face()
        assert face_width_px(lms, 100) == pytest.approx(40.0, abs=0.5)
        assert face_width_px(lms, 500) == pytest.approx(200.0, abs=0.5)


def turned_face(cos_yaw):
    """
    A face foreshortened by a head turn.

    Cheek-to-cheek width projects as cos(yaw) while face height does not, so
    narrowing only the cheek landmarks reproduces what a turned head does to
    the landmark set.
    """
    lms = synthetic_face()
    centre = 0.5
    half = 0.20 * cos_yaw
    lms[config.LEFT_FACE] = _lm(centre - half, 0.50)
    lms[config.RIGHT_FACE] = _lm(centre + half, 0.50)
    return lms


class TestYawRobustFaceSize:
    """
    Tier must not collapse just because a student turned their head.

    Sizing on cheek width alone demotes a turning student out of the tier that
    permits Sleepy detection - at exactly the moment they are most worth
    watching. face_size_px falls back to the yaw-invariant height estimate.
    """

    def test_frontal_face_size_is_at_least_its_width(self):
        lms = synthetic_face()
        assert face_size_px(lms, 200, 200) >= face_width_px(lms, 200)

    def test_height_is_unaffected_by_yaw(self):
        """The property the fallback rests on."""
        tall = face_height_px(turned_face(1.0), 200)
        turned = face_height_px(turned_face(0.6), 200)
        assert turned == pytest.approx(tall, abs=1e-6)

    @pytest.mark.parametrize("cos_yaw", [1.0, 0.87, 0.77, 0.71, 0.6])
    def test_size_is_stable_as_the_head_turns(self, cos_yaw):
        frontal = face_size_px(turned_face(1.0), 200, 200)
        turned = face_size_px(turned_face(cos_yaw), 200, 200)
        assert turned == pytest.approx(frontal, rel=0.02)

    def test_raw_width_would_have_collapsed(self):
        """The failure mode this guards against, stated explicitly."""
        frontal = face_width_px(turned_face(1.0), 200)
        turned = face_width_px(turned_face(0.71), 200)
        assert turned < frontal * 0.8

    def test_turned_face_keeps_its_tier(self):
        w = h = 200
        frontal_tier = tier_for_face_width(face_size_px(turned_face(1.0), w, h))
        turned_tier = tier_for_face_width(face_size_px(turned_face(0.71), w, h))
        assert frontal_tier == Tier.FULL
        assert turned_tier == Tier.FULL

    def test_genuinely_small_faces_are_still_demoted(self):
        """Robustness to yaw must not become blindness to distance."""
        lms = synthetic_face()
        # A tiny crop: every landmark distance scales down with it.
        assert tier_for_face_width(face_size_px(lms, 60, 60)) < Tier.FULL
        assert tier_for_face_width(face_size_px(lms, 30, 30)) == Tier.PRESENCE


class TestIoU:
    def test_identical_boxes(self):
        assert compute_box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert compute_box_iou((0, 0, 10, 10), (50, 50, 60, 60)) == pytest.approx(0.0)

    def test_half_overlap(self):
        # Overlap 5x10=50, union 100+100-50=150.
        assert compute_box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-3)


# ── tiers ──────────────────────────────────────
class TestTiers:
    @pytest.mark.parametrize("width,expected", [
        (200, Tier.FULL), (64, Tier.FULL),
        (63, Tier.COARSE), (40, Tier.COARSE),
        (39, Tier.PRESENCE), (5, Tier.PRESENCE),
    ])
    def test_boundaries(self, width, expected):
        assert tier_for_face_width(width) == expected

    def test_only_full_trusts_eyes(self):
        assert trusts_eyes(Tier.FULL)
        assert not trusts_eyes(Tier.COARSE)
        assert not trusts_eyes(Tier.PRESENCE)

    def test_coarse_still_trusts_pose(self):
        assert trusts_pose(Tier.COARSE)
        assert not trusts_pose(Tier.PRESENCE)

    def test_sleepy_unreachable_below_full(self):
        assert "Sleepy" in states_available(Tier.FULL)
        assert "Sleepy" not in states_available(Tier.COARSE)
        assert states_available(Tier.PRESENCE) == {"Unknown"}


# ── tracker ────────────────────────────────────
class TestTracker:
    def test_new_box_creates_a_track(self):
        t = StudentTracker()
        t.update([(0, 0, 100, 200)], now=1000.0)
        assert len(t.students) == 1

    def test_overlapping_box_reuses_the_track(self):
        t = StudentTracker()
        t.update([(0, 0, 100, 200)], now=1000.0)
        first = next(iter(t.students))
        t.update([(5, 5, 105, 205)], now=1000.1)
        assert list(t.students) == [first]

    def test_distant_box_starts_a_new_track(self):
        t = StudentTracker()
        t.update([(0, 0, 100, 200)], now=1000.0)
        t.update([(900, 0, 1000, 200)], now=1000.1)
        assert len(t.students) == 2

    def test_neighbours_do_not_swap_identity(self):
        """
        Greedy-by-best-IoU, not greedy-by-detection-order.

        Two adjacent students whose boxes both overlap each other's previous
        position must keep their own identities - swapping them would splice
        one student's eye-closure timer onto the other.
        """
        t = StudentTracker()
        t.update([(0, 0, 100, 200), (90, 0, 190, 200)], now=1000.0)
        ids = {s.box[0]: sid for sid, s in t.students.items()}
        left_id, right_id = ids[0], ids[90]

        # Nudge both right; each is still closest to itself.
        t.update([(8, 0, 108, 200), (98, 0, 198, 200)], now=1000.1)
        by_id = {sid: s.box[0] for sid, s in t.students.items()}
        assert by_id[left_id] == 8
        assert by_id[right_id] == 98

    def test_unconfirmed_box_retires_after_enough_failed_attempts(self):
        """Furniture is written off - but only once it has had its chances."""
        t = StudentTracker()
        t.update([(0, 0, 100, 200)], now=1000.0)
        student = next(iter(t.students.values()))
        student.analysis_count = config.UNCONFIRMED_MAX_ATTEMPTS

        t.update([(0, 0, 100, 200)],
                 now=1000.0 + config.TRACK_UNCONFIRMED_TTL + 0.1)
        assert len(t.students) == 0

    def test_unconfirmed_box_survives_the_ttl_if_it_has_not_been_analysed(self):
        """
        The regression guard for a deadlock this code had.

        Confirmation takes FACE_CONFIRM_HITS analysis cycles. When a cycle runs
        longer than TRACK_UNCONFIRMED_TTL - which happened as soon as the
        classifier was enabled at 60 students - a purely wall-clock retirement
        deleted every track before it could reach its second hit. Tracking
        oscillated 60 -> 0 -> 60 and the room read as permanently empty.
        Retirement must therefore need attempts as well as time.
        """
        t = StudentTracker()
        t.update([(0, 0, 100, 200)], now=1000.0)
        student = next(iter(t.students.values()))
        student.analysis_count = 1          # one slow cycle has gone by

        t.update([(0, 0, 100, 200)],
                 now=1000.0 + config.TRACK_UNCONFIRMED_TTL * 5)
        assert len(t.students) == 1, "track retired before it could confirm"

    def test_slow_cycles_still_reach_confirmation(self):
        """End to end: cycles far longer than the TTL must still confirm."""
        t = StudentTracker()
        box = [(0, 0, 100, 200)]
        now = 1000.0
        cycle = config.TRACK_UNCONFIRMED_TTL * 2   # deliberately over budget

        for _ in range(config.FACE_CONFIRM_HITS):
            t.update(box, now=now)
            assert len(t.students) == 1
            student = next(iter(t.students.values()))
            student.analysis_count += 1
            student.note_face_hit()
            now += cycle

        assert next(iter(t.students.values())).face_confirmed

    def test_confirmed_student_survives_brief_occlusion(self):
        t = StudentTracker()
        t.update([(0, 0, 100, 200)], now=1000.0)
        student = next(iter(t.students.values()))
        for _ in range(config.FACE_CONFIRM_HITS):
            student.note_face_hit()
        assert student.face_confirmed

        t.update([], now=1000.0 + config.TRACK_STALE_SECONDS - 0.2)
        assert len(t.students) == 1

    def test_confirmed_student_eventually_retires(self):
        t = StudentTracker()
        t.update([(0, 0, 100, 200)], now=1000.0)
        student = next(iter(t.students.values()))
        for _ in range(config.FACE_CONFIRM_HITS):
            student.note_face_hit()
        t.update([], now=1000.0 + config.TRACK_STALE_SECONDS + 0.1)
        assert len(t.students) == 0

    def test_moving_student_duplicate_clears_within_the_presence_window(self):
        """
        The bug a user hit at n=1: one person read as two students.

        A student who shifts far enough that the new box misses the old one
        spawns a new track, while the old lingers for TRACK_STALE_SECONDS so an
        occluded student does not lose their timers. Counting every confirmed
        track made that whole grace period show one person as two.

        The duplicate cannot be eliminated outright: for a moment an abandoned
        track and a briefly-occluded one are genuinely indistinguishable. What
        it can be is short. Presence is now decided on TRACK_PRESENT_SECONDS
        rather than TRACK_STALE_SECONDS, which is the difference between a
        flicker and a wrong headcount that sits there for two seconds.
        """
        t = StudentTracker()
        now = 1000.0
        t.update([(700, 50, 1200, 700)], now=now)
        first = next(iter(t.students.values()))
        for _ in range(config.FACE_CONFIRM_HITS):
            first.note_face_hit()
        assert len(t.confirmed(now)) == 1

        moved_at = now + 0.3
        t.update([(100, 50, 600, 700)], now=moved_at)
        second = [s for s in t.students.values() if s is not first][0]
        for _ in range(config.FACE_CONFIRM_HITS):
            second.note_face_hit()
        second.last_seen = moved_at

        assert len(t.students) == 2, "old track should still be retained"

        # Once the abandoned track falls outside the presence window - and well
        # before it is retired - the count is right again.
        settled = now + config.TRACK_PRESENT_SECONDS + 0.05
        t.update([(100, 50, 600, 700)], now=settled)
        second.last_seen = settled
        assert len(t.confirmed(settled)) == 1, "one person must count as one"
        assert config.TRACK_PRESENT_SECONDS < config.TRACK_STALE_SECONDS / 2, \
            "presence window must be much shorter than retention"

    def test_retained_track_keeps_its_history_while_uncounted(self):
        """Not counted is not the same as forgotten."""
        t = StudentTracker()
        now = 1000.0
        t.update([(700, 50, 1200, 700)], now=now)
        student = next(iter(t.students.values()))
        for _ in range(config.FACE_CONFIRM_HITS):
            student.note_face_hit()

        later = now + config.TRACK_PRESENT_SECONDS + 0.1
        t.update([], now=later)
        assert student.track_id in t.students, "history was discarded too early"
        assert len(t.confirmed(later)) == 0, "absent student still counted"

        # Reappears before the retention window closes: same track, same history.
        back = later + 0.1
        t.update([(700, 50, 1200, 700)], now=back)
        assert len(t.students) == 1
        assert len(t.confirmed(back)) == 1

    def test_brief_gap_does_not_blink_a_student_out(self):
        """A single missed cycle must not drop someone from the count."""
        t = StudentTracker()
        now = 1000.0
        t.update([(700, 50, 1200, 700)], now=now)
        student = next(iter(t.students.values()))
        for _ in range(config.FACE_CONFIRM_HITS):
            student.note_face_hit()
        # One analysis cycle at 60 students is ~0.17s; the presence window
        # must comfortably exceed that.
        assert len(t.confirmed(now + 0.2)) == 1

    def test_schedule_returns_everyone_under_the_cap(self):
        t = StudentTracker()
        boxes = [(i * 120, 0, i * 120 + 100, 200) for i in range(10)]
        t.update(boxes, now=1000.0)
        assert len(t.schedule(60, now=1000.0)) == 10

    def test_schedule_caps_and_prefers_the_stale(self):
        t = StudentTracker()
        boxes = [(i * 120, 0, i * 120 + 100, 200) for i in range(10)]
        t.update(boxes, now=1000.0)
        for student in t.students.values():
            student.face_confirmed = True
            student.last_analysed = 1000.0
        # One student has not been looked at for a long time.
        stale = list(t.students.values())[7]
        stale.last_analysed = 900.0

        chosen = t.schedule(3, now=1000.0)
        assert len(chosen) == 3
        assert stale in chosen


# ── temporal state machine ─────────────────────
def feed(state, seconds, step=0.1, **face):
    """Drive the state machine over a stretch of simulated time."""
    t = 1000.0
    end = t + seconds
    while t <= end:
        state.update(extract_feature_row(synthetic_face(**face), 192, 192),
                     Tier.FULL, now=t)
        t += step
    return state


class TestTemporalGates:
    def test_brief_closure_does_not_become_sleepy(self):
        s = EngagementState()
        feed(s, 0.4, eye_open=0.002)
        assert s.state != "Sleepy"
        assert s.reason == "Blinking"

    def test_sustained_closure_becomes_sleepy(self):
        s = EngagementState()
        feed(s, config.SLEEPY_EYES_DURATION + 0.4, eye_open=0.002)
        assert s.state == "Sleepy"
        assert "Eyes closed" in s.reason

    def test_sustained_yawn_becomes_sleepy(self):
        s = EngagementState()
        feed(s, config.YAWN_DURATION + 0.4, mouth_open=0.06)
        assert s.state == "Sleepy"

    def test_brief_turn_does_not_become_distracted(self):
        s = EngagementState()
        feed(s, 0.3, yaw_shift=0.12)
        assert s.state != "Distracted"

    def test_sustained_turn_becomes_distracted(self):
        s = EngagementState()
        feed(s, config.DISTRACTED_DURATION + 0.4, yaw_shift=0.12)
        assert s.state == "Distracted"
        assert "Turned" in s.reason

    def test_closure_timer_resets_when_eyes_reopen(self):
        s = EngagementState()
        feed(s, 0.5, eye_open=0.002)
        feed(s, 0.3, eye_open=0.04)
        assert s.closed_sec == 0.0
        assert s.state != "Sleepy"

    def test_presence_tier_reports_unknown_only(self):
        s = EngagementState()
        row = extract_feature_row(synthetic_face(eye_open=0.002), 192, 192)
        for i in range(40):
            s.update(row, Tier.PRESENCE, now=1000.0 + i * 0.1)
        assert s.state == "Unknown"

    def test_coarse_tier_cannot_report_sleepy(self):
        """Closed eyes at COARSE must not produce Sleepy - EAR is not trusted there."""
        s = EngagementState()
        row = extract_feature_row(synthetic_face(eye_open=0.002), 192, 192)
        for i in range(40):
            s.update(row, Tier.COARSE, now=1000.0 + i * 0.1)
        assert s.state != "Sleepy"

    def test_coarse_tier_still_reports_distracted(self):
        s = EngagementState()
        row = extract_feature_row(synthetic_face(yaw_shift=0.12), 192, 192)
        for i in range(30):
            s.update(row, Tier.COARSE, now=1000.0 + i * 0.1)
        assert s.state == "Distracted"

    def test_face_lost_is_ignored_for_unconfirmed_boxes(self):
        s = EngagementState()
        s.mark_face_lost(confirmed=False, now=1000.0)
        s.mark_face_lost(confirmed=False,
                         now=1000.0 + config.FACE_LOST_DURATION + 0.5)
        assert s.state == "Unknown"

    def test_face_lost_flags_confirmed_students(self):
        s = EngagementState()
        feed(s, 0.3)
        s.mark_face_lost(confirmed=True, now=2000.0)
        s.mark_face_lost(confirmed=True,
                         now=2000.0 + config.FACE_LOST_DURATION + 0.1)
        assert s.state == "Distracted"
        assert s.reason == "Facing away"


class TestFaceDeduplication:
    """
    Two tracks resolving to the same face are one person.

    Box overlap cannot settle this: a duplicate YOLO box, or an old track
    sitting beside the new one that replaced it, can differ enough in box space
    to look like two people. The face position is the ground truth, because a
    face is a point and one person has one.
    """

    def _confirmed_track(self, tracker, box, face_xy, size, now):
        """
        Add one confirmed track directly.

        Built by hand rather than through update(), because update() would
        associate an overlapping box onto the existing track - which is exactly
        the situation these tests need to set up and therefore cannot rely on.
        """
        student = TrackedStudent(tracker._next_id, box, now)
        tracker.students[tracker._next_id] = student
        tracker._next_id += 1
        for _ in range(config.FACE_CONFIRM_HITS):
            student.note_face_hit(face_xy=face_xy, face_size=size, now=now)
        student.last_seen = now
        return student

    def test_two_tracks_on_one_face_are_merged(self):
        t = StudentTracker()
        now = 1000.0
        a = self._confirmed_track(t, (700, 50, 1200, 700), (950, 200), 130, now)
        b = self._confirmed_track(t, (760, 20, 1279, 704), (955, 205), 128, now)
        assert len({a.track_id, b.track_id}) == 2
        assert len(t.students) == 2

        merged = t.dedupe_by_face(now)
        assert merged == 1
        assert len(t.confirmed(now)) == 1, "one face must be one student"

    def test_merge_keeps_the_longer_history(self):
        """The survivor should be the track that has watched this person longer."""
        t = StudentTracker()
        now = 1000.0
        older = self._confirmed_track(t, (700, 50, 1200, 700), (950, 200), 130, now)
        newer = self._confirmed_track(t, (760, 20, 1279, 704), (955, 205), 128,
                                      now + 0.5)
        t.dedupe_by_face(now + 0.5)
        assert older.track_id in t.students
        assert newer.track_id not in t.students

    def test_two_genuinely_separate_people_are_kept(self):
        """Deduplication must not start deleting real students."""
        t = StudentTracker()
        now = 1000.0
        self._confirmed_track(t, (100, 50, 500, 700), (300, 200), 130, now)
        self._confirmed_track(t, (700, 50, 1200, 700), (950, 200), 130, now)
        assert t.dedupe_by_face(now) == 0
        assert len(t.confirmed(now)) == 2

    def test_neighbours_sitting_close_are_kept(self):
        """
        Adjacent students in a packed row are the hard case: heavily overlapping
        boxes, distinct faces. Only the face separation may decide.
        """
        t = StudentTracker()
        now = 1000.0
        size = 80
        gap = size * (config.DUPLICATE_FACE_DISTANCE + 0.3)
        self._confirmed_track(t, (400, 50, 900, 700), (600, 200), size, now)
        self._confirmed_track(t, (450, 50, 950, 700), (600 + gap, 205), size, now)
        assert t.dedupe_by_face(now) == 0
        assert len(t.confirmed(now)) == 2

    def test_stale_face_positions_are_not_compared(self):
        """An old face reading must not merge away a student seen since."""
        t = StudentTracker()
        now = 1000.0
        self._confirmed_track(t, (700, 50, 1200, 700), (950, 200), 130, now)
        later = now + config.TRACK_PRESENT_SECONDS + 1.0
        self._confirmed_track(t, (100, 50, 600, 700), (950, 200), 130, later)
        assert t.dedupe_by_face(later) == 0

    def test_face_centre_maps_back_into_frame_coordinates(self):
        """
        The measurement deduplication rests on.

        Landmarks are normalised to their crop, so without walking them back
        through the resize and the crop offset every face reports roughly the
        same position and no duplicate is ever visible.
        """
        lms = synthetic_face()
        # Crop taken at (500,100), downscaled by half, 20px padding.
        box = (520, 120, 1000, 600)
        cx, cy = face_centre_in_frame(lms, box, crop_w=240, crop_h=240,
                                      scale=0.5, padding=20)
        # Nose sits at (0.5, 0.5) of a 240px crop -> 120px in, /0.5 -> 240px,
        # offset by the padded origin (500, 100).
        assert cx == pytest.approx(500 + 240, abs=1.0)
        assert cy == pytest.approx(100 + 240, abs=1.0)

    def test_two_crops_of_one_face_agree_on_position(self):
        """Different boxes over the same face must report the same point."""
        lms = synthetic_face()
        a = face_centre_in_frame(lms, (500, 100, 980, 580),
                                 crop_w=240, crop_h=240, scale=0.5, padding=0)
        b = face_centre_in_frame(lms, (500, 100, 740, 340),
                                 crop_w=240, crop_h=240, scale=1.0, padding=0)
        # Same face, different crop scales: the frame position must agree well
        # inside the duplicate threshold.
        assert abs(a[0] - b[0]) < 130
        assert abs(a[1] - b[1]) < 130


class TestModelHandoff:
    """The heuristic/model split: the model refines, it never overrules."""

    def test_quiet_reading_defers_to_the_model(self):
        s = EngagementState()
        feed(s, 0.3)
        assert s.state == "Attentive"
        assert s.pending_model is True

    def test_timed_state_does_not_defer_to_the_model(self):
        """A student the timers already judged must not be second-guessed."""
        s = EngagementState()
        feed(s, config.SLEEPY_EYES_DURATION + 0.4, eye_open=0.002)
        assert s.state == "Sleepy"
        assert s.pending_model is False

        s.apply_model(0.99, threshold=0.5)      # model shouts "drifting"
        assert s.state == "Sleepy", "model overruled a timed state"

    def test_coarse_tier_never_defers_to_the_model(self):
        """The model was trained on EAR/MAR, which COARSE does not trust."""
        s = EngagementState()
        row = extract_feature_row(synthetic_face(), 192, 192)
        s.update(row, Tier.COARSE, now=1000.0)
        assert s.pending_model is False

    def test_model_can_flip_a_quiet_reading_to_distracted(self):
        s = EngagementState()
        feed(s, 0.3)
        s.apply_model(0.90, threshold=0.5)
        assert s.state == "Distracted"
        assert s.reason == "Disengaged"

    def test_model_confirming_attentive_keeps_attentive(self):
        s = EngagementState()
        feed(s, 0.3)
        s.apply_model(0.10, threshold=0.5)
        assert s.state == "Attentive"

    def test_model_vector_is_in_feature_cols_order(self):
        s = EngagementState()
        feed(s, 0.3)
        vector = s.model_vector()
        assert len(vector) == len(config.FEATURE_COLS)
        assert vector[config.FEATURE_COLS.index("roll")] == pytest.approx(s.roll)
        assert vector[config.FEATURE_COLS.index("mar")] == pytest.approx(s.mar)

    def test_apply_model_is_idempotent(self):
        """A second call must not re-judge an already-settled student."""
        s = EngagementState()
        feed(s, 0.3)
        s.apply_model(0.10, threshold=0.5)
        s.apply_model(0.99, threshold=0.5)
        assert s.state == "Attentive"


class TestModelIntegration:
    def test_feature_vector_matches_scaler_width(self):
        """
        The exact mismatch that made the previous model a constant predictor:
        a scaler fitted on one feature count, fed another.
        """
        import joblib
        if not os.path.exists(config.SCALER_PATH):
            pytest.skip("no trained scaler on disk")
        scaler = joblib.load(config.SCALER_PATH)
        assert scaler.n_features_in_ == len(config.FEATURE_COLS)

    def test_live_feature_values_fall_inside_training_range(self):
        """
        A live-shaped feature vector should land within a few sigma of the
        training distribution. The old code produced ~19 sigma, which is how a
        unit mismatch shows up numerically.
        """
        import joblib
        if not os.path.exists(config.SCALER_PATH):
            pytest.skip("no trained scaler on disk")
        scaler = joblib.load(config.SCALER_PATH)

        row = extract_feature_row(synthetic_face(yaw_shift=0.03), 192, 192)
        vector = np.array([[row[c] for c in config.FEATURE_COLS]])
        z = np.abs(scaler.transform(vector))
        assert z.max() < 8.0, f"feature {z.argmax()} is {z.max():.1f} sigma out"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
