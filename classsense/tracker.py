# classsense/tracker.py
# Identity across frames.
#
# Without this every frame would re-judge strangers, and no temporal signal
# could exist at all - you cannot time a 1.2s eye closure if you do not know
# it is the same pair of eyes. Association is greedy IoU, which is enough for
# seated students: they occupy stable positions and rarely swap places.

import math
import time

from classsense.config import (
    TRACK_IOU_MATCH, TRACK_STALE_SECONDS, TRACK_UNCONFIRMED_TTL,
    UNCONFIRMED_MAX_ATTEMPTS, FACE_CONFIRM_HITS, TRACK_PRESENT_SECONDS,
    DUPLICATE_FACE_DISTANCE, STABLE_AFTER_SECONDS, STABLE_PRIORITY_PENALTY,
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

        # Where this student's face last landed in frame coordinates, and how
        # big it was. Two tracks reporting the same face position are one
        # person - the only reliable way to catch that, since their boxes may
        # legitimately differ.
        self.face_xy = None
        self.face_size = 0.0
        self.face_seen_at = 0.0

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

    def note_face_hit(self, face_xy=None, face_size=0.0, now=None):
        self.face_hit_count += 1
        if self.face_hit_count >= FACE_CONFIRM_HITS:
            self.face_confirmed = True
        if face_xy is not None:
            self.face_xy = face_xy
            self.face_size = face_size
            self.face_seen_at = now or time.time()

    def is_present(self, now=None):
        """
        Seen recently enough to count as in the room.

        Distinct from being retained. A track keeps its history for
        TRACK_STALE_SECONDS so a brief occlusion does not reset a student's
        timers, but it stops being counted as soon as it stops being seen -
        otherwise a student who shifts seats is two students until the old
        track expires.
        """
        now = now or time.time()
        return (now - self.last_seen) <= TRACK_PRESENT_SECONDS

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

    def confirmed(self, now=None):
        """
        The students actually in the room: face-confirmed and seen recently.

        Both conditions matter. Confirmation keeps furniture out; recency keeps
        ghosts out - a track that has been abandoned but not yet retired is
        still holding a student's history and must not also be holding a place
        in the headcount.
        """
        now = now or time.time()
        return {
            sid: s for sid, s in self.students.items()
            if s.face_confirmed and s.is_present(now)
        }

    def dedupe_by_face(self, now=None):
        """
        Merge tracks that resolved to the same face. Returns how many went.

        Two boxes over one person - a duplicate YOLO detection, or an old track
        that has not yet expired next to the new one that replaced it - will
        each crop a region containing that person, each detect the same face,
        and each keep confirming. Nothing in box space distinguishes that from
        two people standing close together; the faces do, because they land on
        the same point.

        The survivor is the track with the longer history, so a student keeps
        the timers they have accumulated rather than restarting them.
        """
        now = now or time.time()
        recent = [
            s for s in self.students.values()
            if s.face_xy is not None
            and (now - s.face_seen_at) <= TRACK_PRESENT_SECONDS
        ]

        merged = set()
        for i, a in enumerate(recent):
            if a.track_id in merged:
                continue
            for b in recent[i + 1:]:
                if b.track_id in merged:
                    continue

                reference = max(a.face_size, b.face_size)
                if reference <= 0:
                    continue
                dx = a.face_xy[0] - b.face_xy[0]
                dy = a.face_xy[1] - b.face_xy[1]
                if math.hypot(dx, dy) > reference * DUPLICATE_FACE_DISTANCE:
                    continue

                # Same face. Keep whichever has watched this person longer.
                keeper, loser = (a, b) if a.created_at <= b.created_at else (b, a)
                merged.add(loser.track_id)
                keeper.last_seen = max(keeper.last_seen, loser.last_seen)
                keeper.face_hit_count += loser.face_hit_count

        for track_id in merged:
            self.students.pop(track_id, None)
        return len(merged)

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

    def counts(self, now=None):
        """Tally of present students by state, for the dashboard."""
        now = now or time.time()
        tally = {"Attentive": 0, "Sleepy": 0, "Distracted": 0, "Unknown": 0}
        for student in self.confirmed(now).values():
            tally[student.state] = tally.get(student.state, 0) + 1
        return tally
