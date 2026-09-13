# classsense/tracker.py
# Identity across frames.
#
# Without this every frame would re-judge strangers, and no temporal signal
# could exist at all - you cannot time a 1.2s eye closure if you do not know
# it is the same pair of eyes. Association is greedy IoU, which is enough for
# seated students: they occupy stable positions and rarely swap places.

import time

from classsense.config import (
    TRACK_IOU_MATCH, TRACK_STALE_SECONDS, TRACK_UNCONFIRMED_TTL,
    UNCONFIRMED_MAX_ATTEMPTS, FACE_CONFIRM_HITS,
    STABLE_AFTER_SECONDS, STABLE_PRIORITY_PENALTY,
)
from classsense.geometry import compute_box_iou
from classsense.states import EngagementState


class TrackedStudent:
    """One person, persisting across frames."""

    def __init__(self, track_id, box, now=None):
        now = now or time.time()
        self.track_id = track_id
        self.box = box
        self.created_at = now
        self.last_seen = now
        self.last_analysed = 0.0

        # A YOLO person box is not yet a student. Chairs, coat stands and
        # posters all draw person boxes at low confidence; requiring repeated
        # face landmark hits before counting one keeps the furniture out of
        # the tally.
        self.face_confirmed = False
        self.face_hit_count = 0
        # Analysis attempts, hit or miss. Retirement counts attempts rather
        # than seconds, so a slow machine cannot starve confirmation.
        self.analysis_count = 0

        self.engagement = EngagementState()

    @property
    def state(self):
        return self.engagement.state

    @property
    def color(self):
        return self.engagement.color

    @property
    def tier(self):
        return self.engagement.tier

    def note_face_hit(self):
        self.face_hit_count += 1
        if self.face_hit_count >= FACE_CONFIRM_HITS:
            self.face_confirmed = True

    def priority(self, now=None):
        """
        How much this student needs looking at, higher first.

        Used only when the cohort is larger than one cycle can cover. Two
        pressures: time since last analysed, so nobody starves; and how settled
        the student is, so a student who has read Attentive for a minute yields
        their slot to one mid-transition.
        """
        now = now or time.time()
        staleness = now - self.last_analysed

        if self.engagement.stable_seconds > STABLE_AFTER_SECONDS:
            staleness *= STABLE_PRIORITY_PENALTY

        # Anything mid-event is urgent: a closure or turn in progress is
        # exactly when a missed sample changes the conclusion.
        if self.engagement.closed_sec > 0 or self.engagement.dist_sec > 0:
            staleness *= 3.0

        # A student not yet confirmed needs hits to resolve either way.
        if not self.face_confirmed:
            staleness *= 2.0

        return staleness


class StudentTracker:
    """Associates detected boxes with persistent students."""

    def __init__(self):
        self.students = {}
        self._next_id = 1

    def update(self, boxes, now=None):
        """
        Match this cycle's boxes onto existing tracks, spawn and retire as needed.

        Greedy by descending IoU rather than first-come: iterating boxes in
        detection order lets an early box claim a track that overlaps a later
        box far better, which swaps two neighbours' identities and scrambles
        both their timers. Sorting all candidate pairs first avoids that.
        """
        now = now or time.time()

        pairs = []
        for box_idx, box in enumerate(boxes):
            for sid, student in self.students.items():
                iou = compute_box_iou(box, student.box)
                if iou >= TRACK_IOU_MATCH:
                    pairs.append((iou, box_idx, sid))
        pairs.sort(reverse=True)

        claimed_boxes = set()
        claimed_ids = set()
        for iou, box_idx, sid in pairs:
            if box_idx in claimed_boxes or sid in claimed_ids:
                continue
            self.students[sid].box = boxes[box_idx]
            self.students[sid].last_seen = now
            claimed_boxes.add(box_idx)
            claimed_ids.add(sid)

        for box_idx, box in enumerate(boxes):
            if box_idx in claimed_boxes:
                continue
            student = TrackedStudent(self._next_id, box, now)
            self.students[self._next_id] = student
            self._next_id += 1

        self._retire(now)
        return self.students

    def _retire(self, now):
        """
        Drop tracks that have gone away.

        A confirmed student gets TRACK_STALE_SECONDS of grace, so a brief
        occlusion by someone walking past does not reset their history.

        An unconfirmed box is written off only once it has had both enough
        attempts and enough time. Requiring both matters: confirmation needs
        FACE_CONFIRM_HITS successive analysis cycles, so a wall clock alone
        would retire every track before it could confirm on any hardware where
        a cycle runs longer than the TTL - and the room would read as
        permanently empty rather than merely slow.
        """
        stale = []
        for sid, s in self.students.items():
            if now - s.last_seen > TRACK_STALE_SECONDS:
                stale.append(sid)
                continue
            if (not s.face_confirmed
                    and s.analysis_count >= UNCONFIRMED_MAX_ATTEMPTS
                    and now - s.created_at > TRACK_UNCONFIRMED_TTL):
                stale.append(sid)

        for sid in stale:
            del self.students[sid]

    def confirmed(self):
        """Only the tracks that have proven they are people."""
        return {sid: s for sid, s in self.students.items() if s.face_confirmed}

    def schedule(self, limit, now=None):
        """
        Choose which students to analyse this cycle.

        On this hardware 60 students fit inside one cycle, so this normally
        returns everyone and the priority ordering is inert. It matters on
        slower machines or in larger rooms, where it decides who gets looked
        at rather than letting a fixed rotation decide.
        """
        now = now or time.time()
        students = list(self.students.values())
        if len(students) <= limit:
            return students
        students.sort(key=lambda s: s.priority(now), reverse=True)
        return students[:limit]

    def counts(self):
        """Tally of confirmed students by state, for the dashboard."""
        tally = {"Attentive": 0, "Sleepy": 0, "Distracted": 0, "Unknown": 0}
        for student in self.confirmed().values():
            tally[student.state] = tally.get(student.state, 0) + 1
        return tally
