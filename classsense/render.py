# classsense/render.py
# Overlay drawing.
#
# The hard part at 60 students is not drawing - it is restraint. Sixty boxes
# each with a state, a confidence and a reason is 180 pieces of text on one
# screen, which is less readable than no annotation at all. So annotation
# density falls as the room fills: in a full room only the students who need
# attention get words, everyone else gets a coloured outline, and the dashboard
# carries the totals.

import cv2

from classsense.config import (
    GREEN, ORANGE, RED, GRAY, YELLOW, WHITE, BLACK, BLUE, PANEL_BG,
    EAR_CLOSED_THRESH, MAR_YAWN_THRESH, YAW_DISTRACTED_THRESH,
    PITCH_RECLINED_THRESH, PITCH_NOD_THRESH, ROLL_RECLINED_THRESH,
)
from classsense.tiers import Tier, TIER_LABELS

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Above this many students, drop to outline-only for the calm majority.
CROWDED_THRESHOLD = 20

# States worth interrupting a crowded view for.
NEEDS_ATTENTION = {"Sleepy", "Distracted"}


def draw_student(frame, student, crowded, show_debug=False):
    """Draw one student's box and however much text the room can afford."""
    x1, y1, x2, y2 = student.box
    color = student.color
    state = student.state

    thickness = 1 if crowded else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # In a crowded room, calm students get the outline and nothing else.
    if crowded and state not in NEEDS_ATTENTION and not show_debug:
        return

    label = f"{state} {student.engagement.confidence:.0%}"
    if state == "Unknown":
        label = "Unknown"

    scale = 0.42 if crowded else 0.55
    weight = 1 if crowded else 2
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, weight)
    tag_top = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, tag_top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5), FONT, scale, WHITE, weight)

    # The reason line is the first thing to go when space is tight.
    if not crowded:
        reason = student.engagement.reason
        (rw, rh), _ = cv2.getTextSize(reason, FONT, 0.42, 1)
        cv2.rectangle(frame, (x1, y2), (x1 + rw + 8, y2 + rh + 8), color, -1)
        cv2.putText(frame, reason, (x1 + 4, y2 + rh + 4), FONT, 0.42, WHITE, 1)

    if show_debug:
        _draw_telemetry(frame, student, x1, y1)


def _draw_telemetry(frame, student, x1, y1):
    """Per-student metrics with the threshold each one is being judged against."""
    e = student.engagement
    tier_note = TIER_LABELS[e.tier]

    if e.tier >= Tier.FULL:
        lines = [
            f"tier {tier_note}",
            f"EAR {e.ear:.2f} {'CLOSED' if e.ear < EAR_CLOSED_THRESH else 'open'}",
            f"MAR {e.mar:.2f} {'YAWN' if e.mar > MAR_YAWN_THRESH else 'ok'}",
        ]
    else:
        lines = [f"tier {tier_note}"]

    if e.tier >= Tier.COARSE:
        lines += [
            f"yaw {e.yaw:+.0f} {'TURN' if abs(e.yaw) > YAW_DISTRACTED_THRESH else 'ok'}",
            f"pit {e.pitch:+.0f} {'REC' if e.pitch < PITCH_RECLINED_THRESH else ('NOD' if e.pitch > PITCH_NOD_THRESH else 'ok')}",
            f"rol {e.roll:+.0f} {'TILT' if abs(e.roll) > ROLL_RECLINED_THRESH else 'ok'}",
        ]
        lines.append(f"close {e.closed_sec:.1f}s dist {e.dist_sec:.1f}s")

    y = y1 + 14
    for line in lines:
        # Black underlay first: yellow on a bright classroom wall is invisible.
        cv2.putText(frame, line, (x1 + 5, y), FONT, 0.35, BLACK, 3)
        cv2.putText(frame, line, (x1 + 5, y), FONT, 0.35, YELLOW, 1)
        y += 13


def draw_dashboard(frame, counts, fps, analysis_fps, tracked, readable,
                   show_debug, cycle_ms=0.0):
    """Summary panel. At 60 students this, not the boxes, is what gets read."""
    h, w = frame.shape[:2]
    total = sum(counts.values())
    attentive = counts.get("Attentive", 0)
    unknown = counts.get("Unknown", 0)

    # Engagement is computed over students we can actually read. Counting the
    # unreadable back row as disengaged would make the number a measure of
    # camera placement rather than of the class.
    engagement = (attentive / readable * 100) if readable else 0.0

    panel_w, panel_h = 300, 250
    px, py = w - panel_w - 12, 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, "ClassSense AI", (px + 12, py + 26), FONT, 0.62, BLUE, 2)

    rows = [
        (f"Students   : {total}", WHITE),
        (f"Attentive  : {attentive}", GREEN),
        (f"Sleepy     : {counts.get('Sleepy', 0)}", ORANGE),
        (f"Distracted : {counts.get('Distracted', 0)}", RED),
        (f"Unreadable : {unknown}", GRAY),
    ]
    y = py + 52
    for text, color in rows:
        cv2.putText(frame, text, (px + 12, y), FONT, 0.46, color, 1)
        y += 21

    # Engagement bar, over readable students only.
    bx, by, bw = px + 12, y + 4, panel_w - 24
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (55, 55, 55), -1)
    fill = int(bw * engagement / 100)
    bar_color = GREEN if engagement >= 70 else YELLOW if engagement >= 40 else RED
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 14), bar_color, -1)

    caption = f"{engagement:.0f}% engaged"
    if unknown:
        caption += f"  (of {readable} readable)"
    cv2.putText(frame, caption, (bx, by + 30), FONT, 0.42, WHITE, 1)

    # Throughput. analysis_fps is the number that actually matters - display
    # fps only says the window is smooth, not that anyone is being assessed.
    perf = f"display {fps:.0f}fps  analysis {analysis_fps:.1f}/s"
    cv2.putText(frame, perf, (px + 12, by + 52), FONT, 0.38, WHITE, 1)
    if show_debug and cycle_ms:
        cv2.putText(frame, f"cycle {cycle_ms:.0f}ms  tracked {tracked}",
                    (px + 12, by + 70), FONT, 0.38, YELLOW, 1)

    hud = "[D] HUD on" if show_debug else "[D] HUD off"
    cv2.putText(frame, f"[Q] quit  [S] snap  {hud}",
                (px + 12, py + panel_h - 10), FONT, 0.38,
                YELLOW if show_debug else WHITE, 1)


def draw_banner(frame, text, color=BLUE):
    """Bottom-left status line, used for startup and warnings."""
    h = frame.shape[0]
    cv2.putText(frame, text, (12, h - 12), FONT, 0.5, BLACK, 3)
    cv2.putText(frame, text, (12, h - 12), FONT, 0.5, color, 1)
