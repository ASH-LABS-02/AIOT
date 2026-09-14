# classsense/tiers.py
# Resolution tiers: how much the pixels actually support saying.
#
# At 1080p a full classroom does not give every face the same number of pixels.
# The front row might land 120px of face width and the back row 40px. Running
# identical analysis on both and printing identical-looking labels would mean
# the label's confidence has nothing to do with its trustworthiness.
#
# So each student carries a tier, and the tier bounds which states they can be
# assigned. A back-row student reads "Unknown" in gray rather than a confident
# "Attentive" derived from noise.

from enum import IntEnum

from classsense.config import (
    TIER_FULL_MIN_WIDTH, TIER_COARSE_MIN_WIDTH,
    GREEN, ORANGE, RED, GRAY,
)


class Tier(IntEnum):
    """Ordered worst to best so comparisons like `tier >= Tier.COARSE` read naturally."""
    PRESENCE = 0   # < 40px face: we know someone is there, nothing more
    COARSE   = 1   # 40-64px: head pose only
    FULL     = 2   # >= 64px: everything, including eye and mouth measures


TIER_LABELS = {
    Tier.PRESENCE: "too far",
    Tier.COARSE:   "pose only",
    Tier.FULL:     "full",
}


def tier_for_face_width(width_px):
    """Map a measured face width in pixels onto a tier."""
    if width_px >= TIER_FULL_MIN_WIDTH:
        return Tier.FULL
    if width_px >= TIER_COARSE_MIN_WIDTH:
        return Tier.COARSE
    return Tier.PRESENCE


def trusts_eyes(tier):
    """
    Whether eye and mouth measures are worth believing at this tier.

    Only FULL. At a 50px face the six eye landmarks span roughly 3px, so a
    single pixel of landmark jitter moves EAR by about 10% - larger than the
    gap between an open and a closed eye. Head pose uses landmarks spanning the
    whole face, so it survives one tier lower.
    """
    return tier >= Tier.FULL


def trusts_pose(tier):
    """Whether head pose is worth believing at this tier."""
    return tier >= Tier.COARSE


def states_available(tier):
    """The set of states this tier can legitimately produce."""
    if tier >= Tier.FULL:
        return {"Attentive", "Sleepy", "Distracted"}
    if tier >= Tier.COARSE:
        return {"Attentive", "Distracted"}
    return {"Unknown"}


STATE_COLORS = {
    "Attentive":   GREEN,
    "Sleepy":      ORANGE,
    "Distracted":  RED,
    "Unknown":     GRAY,
    # Beyond measured capacity: present and counted, deliberately not judged.
    "Unmonitored": GRAY,
}


def color_for_state(state):
    return STATE_COLORS.get(state, GRAY)
