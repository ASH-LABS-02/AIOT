# tests/test_session.py
#
# Session accounting. The arithmetic here becomes a number a teacher acts on,
# so the parts worth pinning down are the ones where a plausible-looking
# implementation would quietly mislead:
#
#   - unreadable time must never count as inattention
#   - time-weighting, not event-counting, must decide who "slept"
#   - coverage must travel beside every score

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from classsense import report_html
from classsense.session import (
    SessionRecorder, MIN_SLEEP_EPISODE, JUDGED_STATES,
)


class Fake:
    """Stands in for a TrackedStudent; the recorder only reads these two."""

    def __init__(self, track_id, state):
        self.track_id = track_id
        self.state = state


def run(script, step=1.0, timeline_interval=1e9, warmup=True):
    """
    Drive a session from a list of per-tick rosters.

    The first observe() can only establish a baseline - there is no previous
    timestamp to attribute an interval to - so by default the first roster is
    replayed once as a warm-up and the script's own ticks all count. That keeps
    the expected numbers in these tests equal to the tick counts rather than
    one less, which would otherwise look like an off-by-one in the accounting.

    timeline_interval defaults to effectively never, so tests that care about
    accounting are not perturbed by timeline sampling.
    """
    rec = SessionRecorder(sample_interval=0.0,
                          timeline_interval=timeline_interval)
    t = rec.started
    ticks = ([script[0]] + list(script)) if (warmup and script) else list(script)
    for roster in ticks:
        t += step
        rec.observe([Fake(i, s) for i, s in roster], now=t)
    rec.finalise(t)
    return rec


class TestTimeAccounting:
    def test_attentiveness_is_share_of_monitored_time(self):
        # 6 ticks attentive, 4 distracted -> 0.6
        script = [[(1, "Attentive")]] * 6 + [[(1, "Distracted")]] * 4
        rec = run(script)
        assert rec.students[1].attentiveness == pytest.approx(0.6, abs=1e-6)

    def test_unreadable_time_is_excluded_not_counted_against(self):
        """
        The central honesty rule. A student we could not see is not a student
        who was inattentive; folding Unknown into the denominator would let
        poor camera placement read as poor engagement.
        """
        script = [[(1, "Attentive")]] * 5 + [[(1, "Unknown")]] * 5
        rec = run(script)
        record = rec.students[1]
        assert record.attentiveness == pytest.approx(1.0)
        assert record.monitored_seconds == pytest.approx(5.0, abs=1e-6)
        assert record.present_seconds == pytest.approx(10.0, abs=1e-6)

    def test_unmonitored_time_is_excluded_too(self):
        script = [[(1, "Attentive")]] * 5 + [[(1, "Unmonitored")]] * 15
        rec = run(script)
        assert rec.students[1].attentiveness == pytest.approx(1.0)

    def test_never_readable_student_scores_none_not_zero(self):
        """
        None, never 0.0. Zero would rank an unseen student alongside a genuinely
        inattentive one, and every average downstream would inherit that.
        """
        rec = run([[(1, "Unknown")]] * 10)
        assert rec.students[1].attentiveness is None

    def test_coverage_reports_how_much_was_readable(self):
        script = [[(1, "Attentive")]] * 3 + [[(1, "Unknown")]] * 7
        rec = run(script)
        assert rec.students[1].coverage == pytest.approx(0.3, abs=1e-6)

    def test_judged_states_exclude_the_absence_states(self):
        assert "Unknown" not in JUDGED_STATES
        assert "Unmonitored" not in JUDGED_STATES


class TestSleepEpisodes:
    def test_a_sustained_run_is_an_episode(self):
        script = [[(1, "Attentive")]] * 2 + [[(1, "Sleepy")]] * 6 \
                 + [[(1, "Attentive")]] * 2
        rec = run(script)
        record = rec.students[1]
        assert record.slept
        assert len(record.sleep_episodes) == 1
        assert record.sleep_seconds == pytest.approx(6.0, abs=1e-6)

    def test_a_flicker_shorter_than_the_floor_is_not_an_episode(self):
        """
        The state machine already needs 1.2s of closed eyes to enter Sleepy, so
        anything briefer than the floor is a state bouncing at its boundary -
        not something to put in front of a teacher as "this student slept".
        """
        ticks = int(MIN_SLEEP_EPISODE) - 1
        script = [[(1, "Attentive")]] + [[(1, "Sleepy")]] * ticks \
                 + [[(1, "Attentive")]] * 3
        rec = run(script)
        assert rec.students[1].slept is False

    def test_separate_runs_are_separate_episodes(self):
        script = ([[(1, "Sleepy")]] * 4 + [[(1, "Attentive")]] * 3
                  + [[(1, "Sleepy")]] * 5)
        rec = run(script)
        assert len(rec.students[1].sleep_episodes) == 2

    def test_longest_episode_is_reported(self):
        script = ([[(1, "Sleepy")]] * 3 + [[(1, "Attentive")]]
                  + [[(1, "Sleepy")]] * 9)
        rec = run(script)
        assert rec.students[1].longest_sleep == pytest.approx(9.0, abs=1e-6)

    def test_sleeping_at_the_end_still_closes_the_episode(self):
        """finalise() must not drop a run that was still open when time ran out."""
        rec = run([[(1, "Attentive")]] + [[(1, "Sleepy")]] * 8)
        assert rec.students[1].slept
        assert rec.students[1].sleep_seconds == pytest.approx(8.0, abs=1e-6)


class TestHeadlineNumbers:
    def _class(self):
        # Three students: one attentive, one sleeps a long stretch, one drifts.
        script = []
        for tick in range(20):
            script.append([
                (1, "Attentive"),
                (2, "Sleepy" if 5 <= tick < 15 else "Attentive"),
                (3, "Distracted" if tick % 2 else "Attentive"),
            ])
        return run(script).report()

    def test_counts_students_seen(self):
        assert self._class()["headline"]["students_seen"] == 3

    def test_pct_never_slept_counts_students(self):
        """Two of three never slept."""
        assert self._class()["headline"]["pct_never_slept"] == pytest.approx(66.7, abs=0.2)

    def test_pct_time_not_sleeping_measures_time(self):
        head = self._class()["headline"]
        # 10 sleepy seconds out of 60 monitored student-seconds.
        assert head["pct_time_not_sleeping"] == pytest.approx(83.3, abs=0.3)

    def test_the_two_not_sleeping_figures_are_different_questions(self):
        """
        A headcount and a time-share answer different things and can diverge
        sharply - one long sleeper barely moves the headcount but dominates the
        time. Reporting only one would be choosing an answer silently.
        """
        head = self._class()["headline"]
        assert head["pct_never_slept"] != head["pct_time_not_sleeping"]

    def test_both_means_are_reported(self):
        head = self._class()["headline"]
        assert head["mean_attentiveness"] is not None
        assert head["weighted_attentiveness"] is not None

    def test_students_are_ordered_lowest_attentiveness_first(self):
        students = self._class()["students"]
        scores = [s["attentiveness"] for s in students]
        assert scores == sorted(scores)

    def test_empty_session_produces_a_usable_report(self):
        """An empty room must render, not raise."""
        report = run([]).report()
        assert report["headline"]["students_seen"] == 0
        assert report["headline"]["mean_attentiveness"] is None
        report_html.render(report)          # must not raise


class TestTimeline:
    def test_samples_at_the_configured_interval(self):
        rec = run([[(1, "Attentive")]] * 30, step=1.0, timeline_interval=10.0)
        assert 2 <= len(rec.timeline) <= 4

    def test_engagement_is_over_readable_students_only(self):
        rec = run([[(1, "Attentive"), (2, "Unknown")]] * 12,
                  step=1.0, timeline_interval=5.0)
        point = rec.timeline[0]
        assert point["engagement"] == pytest.approx(1.0)
        assert point["present"] == 2

    def test_engagement_is_none_when_nobody_is_readable(self):
        rec = run([[(1, "Unknown"), (2, "Unmonitored")]] * 12,
                  step=1.0, timeline_interval=5.0)
        assert rec.timeline[0]["engagement"] is None


class TestReportRendering:
    def _report(self):
        script = []
        for tick in range(24):
            script.append([
                (1, "Attentive"),
                (2, "Sleepy" if tick > 12 else "Attentive"),
                (3, "Unknown"),
            ])
        return run(script, step=1.0, timeline_interval=4.0).report()

    def test_renders_self_contained_html(self):
        """
        No CDN, no external stylesheet, no script src: a Pi in a classroom may
        have no route to the internet, and the file must still open months
        later off a laptop.
        """
        page = report_html.render(self._report())
        assert page.lstrip().startswith("<!doctype html>")
        assert "<svg" in page
        for forbidden in ("http://", "https://", "src=", "@import"):
            assert forbidden not in page, f"external reference: {forbidden}"

    def test_legend_only_lists_states_that_are_drawn(self):
        """
        A state can hold real time and never land on a timeline sample. The
        legend must follow the chart, or it points at a colour the reader
        cannot find.
        """
        report = self._report()
        shown = report_html.states_in_timeline(report["timeline"])
        for state in shown:
            assert any(p["counts"].get(state) for p in report["timeline"])

    def test_states_never_rely_on_colour_alone(self):
        """Status colour always ships with its label."""
        page = report_html.render(self._report())
        for state in ("Attentive", "Sleepy"):
            assert state in page

    def test_dark_mode_is_declared_both_ways(self):
        page = report_html.render(self._report())
        assert "prefers-color-scheme: dark" in page
        assert '[data-theme="dark"]' in page

    def test_no_unresolved_format_placeholders(self):
        page = report_html.render(self._report())
        assert "{sample_note}" not in page
        assert "{}" not in page

    def test_renders_when_the_timeline_is_too_short_to_plot(self):
        report = run([[(1, "Attentive")]] * 2, timeline_interval=1e9).report()
        page = report_html.render(report)
        assert "Not enough samples" in page


class TestDurationFormatting:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "0s"), (45, "45s"), (60, "1m 00s"),
        (125, "2m 05s"), (3600, "1h 00m"), (3725, "1h 02m"),
    ])
    def test_formats(self, seconds, expected):
        assert report_html.fmt_duration(seconds) == expected

    def test_none_reads_as_zero(self):
        assert report_html.fmt_duration(None) == "0s"

    def test_pct_of_none_is_an_em_dash(self):
        assert report_html.pct(None) == "—"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
