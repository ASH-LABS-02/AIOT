# classsense/session.py
# Session recording: what actually happened over a lesson.
#
# The live view answers "what is happening now". This answers "how did the
# lesson go", which is a different question and needs different bookkeeping -
# time in each state per student, when people drifted, who slept and for how
# long.
#
# Everything here is time-weighted from periodic samples rather than counted in
# events. A student who is Sleepy for 40 seconds and one who blinks into it for
# half a second are not the same thing, and counting state *changes* would rank
# them equally.
#
# The honesty problem this has to solve
# -------------------------------------
# A student can be Unknown (too far to read) or Unmonitored (beyond capacity)
# for most of a lesson. Averaging their "attentiveness" over wall-clock time
# would quietly score them on the minority of it we could actually see, and a
# student watched for 8% of the lesson would appear in the report looking
# exactly as authoritative as one watched throughout. So monitored time is
# tracked separately from present time, every score is over monitored time, and
# coverage is reported beside every score.

import json
import os
import time
from collections import Counter

from classsense import config

# States that represent a real reading. Unknown and Unmonitored are absence of
# information, not a finding, and must never land in a denominator.
JUDGED_STATES = ("Attentive", "Sleepy", "Distracted")
ALL_STATES = JUDGED_STATES + ("Unknown", "Unmonitored")

# A Sleepy run shorter than this is treated as noise rather than a sleep
# episode. The state machine already requires 1.2s of closed eyes to enter
# Sleepy, so anything this brief is a state flickering at the boundary.
MIN_SLEEP_EPISODE = 2.0


class StudentRecord:
    """Time accounting for one student across the session."""

    def __init__(self, track_id, first_seen):
        self.track_id = track_id
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.present_seconds = 0.0
        self.state_seconds = {s: 0.0 for s in ALL_STATES}

        self.sleep_episodes = []          # (start, duration)
        self._sleep_started = None
        self._sleep_last = None           # time of the most recent Sleepy sample

    # ── accumulation ───────────────────────────
    def observe(self, state, dt, now):
        """Attribute dt of the session to this student's current state."""
        self.last_seen = now
        self.present_seconds += dt
        if state not in self.state_seconds:
            self.state_seconds[state] = 0.0
        self.state_seconds[state] += dt

        if state == "Sleepy":
            if self._sleep_started is None:
                # Each sample accounts for the interval ending at `now`, so an
                # episode begins where that interval begins.
                self._sleep_started = now - dt
            self._sleep_last = now
        elif self._sleep_started is not None:
            self._close_sleep_episode()

    def _close_sleep_episode(self):
        """
        End the episode at the last Sleepy sample, not at the sample that ended it.

        Closing at the current time would add one whole sampling interval - the
        one in which the student was already awake - to every episode. At a 1s
        sample that silently inflates every reported sleep duration, and the
        duration is the number a teacher would act on.
        """
        duration = self._sleep_last - self._sleep_started
        if duration >= MIN_SLEEP_EPISODE:
            self.sleep_episodes.append((self._sleep_started, duration))
        self._sleep_started = None
        self._sleep_last = None

    def finalise(self, now):
        if self._sleep_started is not None:
            self._close_sleep_episode()

    # ── derived measures ───────────────────────
    @property
    def monitored_seconds(self):
        """Time we could actually read this student. Every score's denominator."""
        return sum(self.state_seconds[s] for s in JUDGED_STATES)

    @property
    def coverage(self):
        """
        Fraction of their time in the room that was readable.

        Reported beside every score, because a 90% attentiveness over 8% of the
        lesson is not the same claim as 90% over all of it, and nothing in the
        number itself says which one you are looking at.
        """
        if self.present_seconds <= 0:
            return 0.0
        return self.monitored_seconds / self.present_seconds

    @property
    def attentiveness(self):
        """
        Share of monitored time spent Attentive, 0-1, or None if never read.

        None rather than 0.0 on purpose: a student we could not see is not a
        student who was inattentive, and collapsing the two would let poor
        camera placement read as poor engagement.
        """
        monitored = self.monitored_seconds
        if monitored <= 0:
            return None
        return self.state_seconds["Attentive"] / monitored

    @property
    def slept(self):
        return bool(self.sleep_episodes)

    @property
    def sleep_seconds(self):
        return sum(d for _, d in self.sleep_episodes)

    @property
    def longest_sleep(self):
        return max((d for _, d in self.sleep_episodes), default=0.0)

    def as_dict(self, session_start):
        return {
            "student": self.track_id,
            "first_seen_s": round(self.first_seen - session_start, 1),
            "present_s": round(self.present_seconds, 1),
            "monitored_s": round(self.monitored_seconds, 1),
            "coverage": round(self.coverage, 3),
            "attentiveness": (None if self.attentiveness is None
                              else round(self.attentiveness, 3)),
            "seconds": {s: round(v, 1) for s, v in self.state_seconds.items()},
            "slept": self.slept,
            "sleep_episodes": len(self.sleep_episodes),
            "sleep_s": round(self.sleep_seconds, 1),
            "longest_sleep_s": round(self.longest_sleep, 1),
        }


class SessionRecorder:
    """
    Samples the live tracker and builds the session report.

    Sampling rather than event-listening keeps this decoupled from the
    pipeline: it reads the same snapshot the renderer does, so it cannot
    perturb analysis or hold its lock.
    """

    def __init__(self, sample_interval=1.0, timeline_interval=15.0):
        self.started = time.time()
        self.ended = None
        self.sample_interval = sample_interval
        self.timeline_interval = timeline_interval

        self.students = {}
        self._last_sample = None
        self._last_timeline = self.started
        self.timeline = []
        self.samples = 0
        self.peak_present = 0

    @property
    def duration(self):
        return (self.ended or time.time()) - self.started

    def observe(self, students, now=None):
        """
        Fold one snapshot of the tracker into the session.

        `students` is the list from AnalysisWorker.snapshot(); only confirmed,
        present students appear in it.
        """
        now = now or time.time()
        if self._last_sample is None:
            self._last_sample = now
            return

        dt = now - self._last_sample
        if dt < self.sample_interval:
            return
        self._last_sample = now
        self.samples += 1
        self.peak_present = max(self.peak_present, len(students))

        for student in students:
            record = self.students.get(student.track_id)
            if record is None:
                record = StudentRecord(student.track_id, now)
                self.students[student.track_id] = record
            record.observe(student.state, dt, now)

        if now - self._last_timeline >= self.timeline_interval:
            self._last_timeline = now
            self.timeline.append(self._timeline_point(students, now))

    def _timeline_point(self, students, now):
        counts = Counter(s.state for s in students)
        judged = sum(counts.get(s, 0) for s in JUDGED_STATES)
        return {
            "t": round(now - self.started, 1),
            "present": len(students),
            "counts": {s: counts.get(s, 0) for s in ALL_STATES},
            # Engagement over readable students only, matching the live view.
            "engagement": (round(counts.get("Attentive", 0) / judged, 3)
                           if judged else None),
        }

    def finalise(self, now=None):
        now = now or time.time()
        self.ended = now
        for record in self.students.values():
            record.finalise(now)

    # ── the report ─────────────────────────────
    def report(self):
        """Everything the dashboard and the written report are built from."""
        now = self.ended or time.time()
        records = list(self.students.values())
        readable = [r for r in records if r.monitored_seconds > 0]

        total_monitored = sum(r.monitored_seconds for r in records)
        state_totals = {
            s: sum(r.state_seconds.get(s, 0.0) for r in records)
            for s in ALL_STATES
        }

        scored = [r.attentiveness for r in readable]
        # Two means, because they answer different questions. The unweighted one
        # treats every student equally, which is what a teacher means by "how
        # was the class". The weighted one is dominated by whoever was on camera
        # longest, which is what the raw time actually supports. Publishing only
        # one of them would be picking an answer without saying so.
        mean_attentiveness = (sum(scored) / len(scored)) if scored else None
        weighted_attentiveness = (
            state_totals["Attentive"] / total_monitored
            if total_monitored > 0 else None
        )

        slept = [r for r in records if r.slept]
        sleep_time = state_totals["Sleepy"]

        return {
            "session": {
                "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(self.started)),
                "ended": time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(now)),
                "duration_s": round(self.duration, 1),
                "samples": self.samples,
            },
            "headline": {
                "students_seen": len(records),
                "peak_present": self.peak_present,
                "students_readable": len(readable),
                # The two framings of "not sleeping", both reported because
                # they can differ sharply: one long sleeper moves the time
                # figure far more than the headcount figure.
                "pct_never_slept": (
                    round(100.0 * (len(records) - len(slept)) / len(records), 1)
                    if records else None
                ),
                "pct_time_not_sleeping": (
                    round(100.0 * (1 - sleep_time / total_monitored), 1)
                    if total_monitored > 0 else None
                ),
                "mean_attentiveness": (None if mean_attentiveness is None
                                       else round(mean_attentiveness, 3)),
                "weighted_attentiveness": (None if weighted_attentiveness is None
                                           else round(weighted_attentiveness, 3)),
                "students_slept": len(slept),
                "sleep_episodes": sum(len(r.sleep_episodes) for r in records),
                "total_sleep_s": round(sleep_time, 1),
                "longest_sleep_s": round(
                    max((r.longest_sleep for r in records), default=0.0), 1
                ),
            },
            "coverage": {
                "monitored_s": round(total_monitored, 1),
                "present_s": round(sum(r.present_seconds for r in records), 1),
                "mean_coverage": (
                    round(sum(r.coverage for r in records) / len(records), 3)
                    if records else None
                ),
                "unreadable_s": round(
                    state_totals["Unknown"] + state_totals["Unmonitored"], 1
                ),
            },
            "state_seconds": {s: round(v, 1) for s, v in state_totals.items()},
            "timeline": self.timeline,
            "students": sorted(
                (r.as_dict(self.started) for r in records),
                key=lambda d: (d["attentiveness"] is None,
                               d["attentiveness"] if d["attentiveness"] is not None else 0),
            ),
        }

    def save(self, directory=None, stem=None):
        """Write the report as JSON. Returns the path."""
        directory = directory or os.path.join(config.DATA_DIR, "reports")
        os.makedirs(directory, exist_ok=True)
        stem = stem or time.strftime("session_%Y%m%d_%H%M%S",
                                     time.localtime(self.started))
        path = os.path.join(directory, f"{stem}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.report(), fh, indent=2)
        return path
