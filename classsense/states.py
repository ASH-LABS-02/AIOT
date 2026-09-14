# classsense/states.py
# The temporal state machine for one student.
#
# The central idea is that none of these states are instantaneous properties.
# A blink and a microsleep look identical in a single frame and differ only in
# duration; a glance at a neighbour and a turned back differ only in duration.
# So every signal is gated on wall-clock time, and a state changes only once
# its evidence has persisted.
#
# Wall-clock rather than frame counts, because the analysis rate moves with how
# many students are in the room - 5 students and 60 students give very
# different frame intervals but the same seconds.

import time
from collections import deque

import numpy as np

from classsense.config import (
    EAR_CLOSED_THRESH, MAR_YAWN_THRESH,
    YAW_DISTRACTED_THRESH, ROLL_RECLINED_THRESH,
    PITCH_RECLINED_THRESH, PITCH_NOD_THRESH,
    SLEEPY_EYES_DURATION, YAWN_DURATION, DISTRACTED_DURATION,
    FACE_LOST_DURATION, SMOOTHING_WINDOW, FEATURE_COLS,
)
from classsense.tiers import Tier, trusts_eyes, trusts_pose, color_for_state


class EngagementState:
    """
    Temporal state for a single tracked person.

    Owns no rendering and no tracking - it takes feature dicts in and exposes a
    state, a reason, and a confidence. That makes it testable by feeding it a
    scripted sequence of features and clock values.
    """

    def __init__(self):
        self.state = "Unknown"
        self.reason = "Acquiring"
        self.confidence = 0.0
        self.tier = Tier.PRESENCE

        # Rolling means. Raw landmark output is jittery frame to frame; the
        # thresholds are set against the smoothed value.
        self.ear_hist   = deque(maxlen=SMOOTHING_WINDOW)
        self.mar_hist   = deque(maxlen=SMOOTHING_WINDOW)
        self.yaw_hist   = deque(maxlen=SMOOTHING_WINDOW)
        self.pitch_hist = deque(maxlen=SMOOTHING_WINDOW)
        self.roll_hist  = deque(maxlen=SMOOTHING_WINDOW)

        # When each condition started, or None if it is not currently true.
        self.eye_closed_start = None
        self.yawn_start       = None
        self.distracted_start = None
        self.face_lost_start  = None

        self.closed_sec = 0.0
        self.yawn_sec   = 0.0
        self.dist_sec   = 0.0

        # Latest smoothed values, for the debug HUD.
        self.ear = 0.0
        self.mar = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.state_since = time.time()
        self._last_state = None

        # Set when the heuristics found nothing conclusive and the classifier
        # would be worth asking. The pipeline collects everyone in this
        # condition and scores them in a single batched call - 60 separate
        # predict_proba calls on a 300-tree forest measure 3.7s against 56ms
        # batched, which is the difference between a working cycle and one
        # slow enough to break tracking.
        self.pending_model = False

    # ── helpers ────────────────────────────────
    @property
    def color(self):
        return color_for_state(self.state)

    @property
    def stable_seconds(self):
        """How long the current state has held. Drives scheduling priority."""
        return time.time() - self.state_since

    def _set_state(self, state, reason, confidence):
        if state != self._last_state:
            self.state_since = time.time()
            self._last_state = state
        self.state = state
        self.reason = reason
        self.confidence = float(confidence)

    def _hold_timer(self, condition, start_attr, now):
        """
        Run one condition's stopwatch. Returns how long it has been true.

        Starting the clock on the first true reading and clearing it on the
        first false one is what makes a 200ms blink and a 2s microsleep
        distinguishable with the same threshold.
        """
        start = getattr(self, start_attr)
        if condition:
            if start is None:
                start = now
                setattr(self, start_attr, start)
            return now - start
        setattr(self, start_attr, None)
        return 0.0

    # ── updates ────────────────────────────────
    def update(self, features, tier, now=None):
        """
        Fold one landmark reading into the state and settle everything the
        heuristics can settle.

        `features` is a dict from geometry.extract_feature_row. `tier` bounds
        what may be concluded from it. If the heuristics reach no conclusion,
        `pending_model` is left set and the caller may follow up with
        apply_model().
        """
        now = now or time.time()
        self.tier = tier
        self.face_lost_start = None
        self.pending_model = False

        self.ear_hist.append(features["avg_ear"])
        self.mar_hist.append(features["mar"])
        self.yaw_hist.append(features["yaw"])
        self.pitch_hist.append(features["pitch"])
        self.roll_hist.append(features["roll"])

        self.ear   = float(np.mean(self.ear_hist))
        self.mar   = float(np.mean(self.mar_hist))
        self.yaw   = float(np.mean(self.yaw_hist))
        self.pitch = float(np.mean(self.pitch_hist))
        self.roll  = float(np.mean(self.roll_hist))

        eyes_ok = trusts_eyes(tier)
        pose_ok = trusts_pose(tier)

        # Eye and mouth timers only run where the pixels support them.
        # Gating on the instantaneous value, not the smoothed one, keeps the
        # onset of a closure crisp - smoothing would delay the start by up to
        # SMOOTHING_WINDOW samples and eat into the measured duration.
        self.closed_sec = self._hold_timer(
            eyes_ok and features["avg_ear"] < EAR_CLOSED_THRESH,
            "eye_closed_start", now,
        )
        self.yawn_sec = self._hold_timer(
            eyes_ok and features["mar"] > MAR_YAWN_THRESH,
            "yawn_start", now,
        )

        reasons = []
        if pose_ok:
            turned   = abs(self.yaw) > YAW_DISTRACTED_THRESH
            reclined = self.pitch < PITCH_RECLINED_THRESH
            tilted   = abs(self.roll) > ROLL_RECLINED_THRESH
            # Head down with eyes open is looking at a lap or a phone. Head down
            # with eyes closed is nodding off, which the Sleepy branch claims -
            # so exclude that case here rather than have both fire.
            head_down = self.pitch > PITCH_NOD_THRESH and self.closed_sec < 0.3

            if turned:    reasons.append(f"Turned {self.yaw:+.0f}°")
            if reclined:  reasons.append(f"Reclined {self.pitch:+.0f}°")
            if tilted:    reasons.append(f"Tilted {self.roll:+.0f}°")
            if head_down: reasons.append("Head down")

            self.dist_sec = self._hold_timer(
                turned or reclined or tilted or head_down,
                "distracted_start", now,
            )
        else:
            self.dist_sec = self._hold_timer(False, "distracted_start", now)

        self._arbitrate(reasons, tier)

    def _arbitrate(self, dist_reasons, tier):
        """
        Decide the state from the timers, most serious signal first.

        Order matters: a student asleep with their head on the desk trips both
        the eye-closure and the head-tilt conditions, and Sleepy is the more
        useful thing to report.
        """
        # Below the resolution floor there is nothing to say beyond presence.
        if tier <= Tier.PRESENCE:
            self._set_state("Unknown", "Too far to read", 0.0)
            return

        # 1. Sleepy - only reachable at FULL tier, since it rests on EAR.
        if trusts_eyes(tier):
            if self.closed_sec >= SLEEPY_EYES_DURATION:
                self._set_state(
                    "Sleepy", f"Eyes closed {self.closed_sec:.1f}s",
                    min(0.99, 0.75 + self.closed_sec * 0.1),
                )
                return
            if self.yawn_sec >= YAWN_DURATION:
                self._set_state("Sleepy", f"Yawning {self.yawn_sec:.1f}s", 0.90)
                return
            if self.closed_sec >= 0.8 and self.pitch > PITCH_NOD_THRESH:
                self._set_state("Sleepy", "Nodding off", 0.88)
                return

        # 2. Distracted - sustained postural evidence.
        if self.dist_sec >= DISTRACTED_DURATION:
            self._set_state(
                "Distracted", ", ".join(dist_reasons) or "Turned away",
                min(0.98, 0.70 + self.dist_sec * 0.1),
            )
            return

        # 3. A blink in progress. Hold Attentive rather than flicker; this is
        #    the branch that stops the label strobing on every natural blink.
        if self.closed_sec > 0.0:
            self._set_state("Attentive", "Blinking", 0.85)
            return

        # 4. Nothing overt. Settle on Attentive, and flag that the classifier
        #    could refine this if one is in use. Committing to a usable state
        #    here rather than waiting means a missing, refused or failed model
        #    costs nothing - the state is already correct-by-default.
        self._set_state("Attentive", "Posture OK", 0.75)
        self.pending_model = trusts_eyes(tier)

    def model_vector(self):
        """
        This student's features in FEATURE_COLS order, for a batched predict.

        Assembled by name then ordered by the config list, so adding or
        reordering a feature cannot silently misalign the vector against the
        scaler that was fitted on it.

        left_ear and right_ear are not smoothed separately - the state keeps
        one averaged EAR - so the smoothed average stands in for both.
        Training averages all three over a clip, so the two paths agree.
        """
        values = {
            "left_ear":  self.ear,
            "right_ear": self.ear,
            "avg_ear":   self.ear,
            "mar":       self.mar,
            "yaw":       self.yaw,
            "pitch":     self.pitch,
            "roll":      self.roll,
        }
        return [values[c] for c in FEATURE_COLS]

    def apply_model(self, drift_proba, threshold=0.5):
        """
        Let the classifier revise an inconclusive reading.

        Only ever called on a student the heuristics left at `pending_model`,
        so the model refines a quiet case and never overrules a timed one.
        """
        if not self.pending_model:
            return
        self.pending_model = False

        if drift_proba >= threshold:
            self._set_state("Distracted", "Disengaged", float(drift_proba))
        else:
            self._set_state("Attentive", "Engaged", float(1.0 - drift_proba))

    def mark_unmonitored(self):
        """
        Present, but beyond what this machine can watch properly.

        The alternative - rotating everyone through at whatever rate the
        hardware manages - produces states that look identical to trustworthy
        ones while being sampled too rarely to catch the events they name. A
        student examined every two seconds can sleep through a whole lesson
        reading "Attentive". Saying "Unmonitored" is the honest output, and it
        tells the operator to add hardware or narrow the camera rather than
        quietly believing a number.
        """
        self.pending_model = False
        self._set_state("Unmonitored", "Beyond capacity", 0.0)

    def mark_face_lost(self, confirmed, now=None):
        """
        No landmarks this cycle.

        For a confirmed student this means they have turned away far enough
        that no face is visible, which is itself informative once sustained.
        For a box that never had a confirmed face it means nothing - it is
        probably a chair - so nothing is concluded.
        """
        now = now or time.time()
        if not confirmed:
            return

        if self.face_lost_start is None:
            self.face_lost_start = now

        if now - self.face_lost_start >= FACE_LOST_DURATION:
            self._set_state("Distracted", "Facing away", 0.85)
