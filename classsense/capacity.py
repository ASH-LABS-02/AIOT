# classsense/capacity.py
# How many students this machine can actually watch properly.
#
# The project was tuned on a 20-core desktop. A Raspberry Pi 5 has four slower
# cores, so every constant derived from that machine is wrong there - and wrong
# in the dangerous direction, because the failure is silent: the pipeline still
# runs, still draws boxes, still prints an engagement percentage, and simply
# stops sampling often enough for any of it to mean anything.
#
# So capacity is measured rather than assumed, and the limit is expressed in
# terms of fidelity rather than throughput.
#
# The binding constraint is not frames per second. It is that a temporal gate
# must be sampled several times inside its own window. SLEEPY_EYES_DURATION is
# 1.2s and DISTRACTED_DURATION is 0.8s; if a student is only examined every
# 1.5s, a 1.2s eye closure can fall entirely between two samples and a sleeping
# student reads as attentive. That is worse than reporting nothing.

import json
import os
import time

from classsense import config

# Frame size the resolution ceiling assumes when the caller does not say.
CAPTURE_W_DEFAULT = config.CAPTURE_WIDTH
CAPTURE_H_DEFAULT = config.CAPTURE_HEIGHT


class Capacity:
    """What this machine measured, and what follows from it."""

    def __init__(self, per_face_ms, detect_ms, cores, pool_size,
                 yolo_width, detect_every, samples_per_gate, source="measured"):
        self.per_face_ms = per_face_ms
        self.detect_ms = detect_ms
        self.cores = cores
        self.pool_size = pool_size
        self.yolo_width = yolo_width
        self.detect_every = detect_every
        self.samples_per_gate = samples_per_gate
        self.source = source

    # ── the fidelity budget ────────────────────
    @property
    def shortest_gate(self):
        """
        The tightest temporal window any state depends on.

        Whichever gate is shortest sets the sampling requirement for all of
        them, because missing it is what makes a state unreliable.
        """
        return min(config.SLEEPY_EYES_DURATION, config.DISTRACTED_DURATION)

    @property
    def max_refresh_seconds(self):
        """Slowest refresh at which every state still holds."""
        return self.shortest_gate / self.samples_per_gate

    @property
    def amortised_detect_ms(self):
        return self.detect_ms / max(1, self.detect_every)

    def cycle_ms_for(self, students):
        """Predicted mean cycle time at this many students."""
        return self.amortised_detect_ms + students * self.per_face_ms

    def compute_ceiling(self):
        """
        The largest cohort that still meets the fidelity budget.

        Zero means this machine cannot honour the budget even for one student -
        detection alone already costs more than the whole budget - which is a
        real answer and better than a number that quietly does not work.
        """
        budget_ms = self.max_refresh_seconds * 1000.0
        spare = budget_ms - self.amortised_detect_ms
        if spare <= 0 or self.per_face_ms <= 0:
            return 0
        return int(spare // self.per_face_ms)

    def resolution_ceiling(self, frame_w=None, frame_h=None, tier_px=None):
        """
        The largest cohort the camera can actually resolve.

        The second ceiling, and the one that bites first on a wide shot.
        Compute capacity says how many faces can be processed; it says nothing
        about whether the pixels exist to process. A 1080p frame holding 150
        students gives each a face about 25px wide, which is below even the
        COARSE floor - every one of them would read Unknown.

        Modelled by tiling: N students across a WxH frame get roughly
        sqrt(W*H/N) of frame per person, of which a face is about a third.
        Rough, but it is the difference between a capacity figure that means
        something and one that is purely arithmetic.
        """
        frame_w = frame_w or CAPTURE_W_DEFAULT
        frame_h = frame_h or CAPTURE_H_DEFAULT
        tier_px = tier_px or config.TIER_FULL_MIN_WIDTH
        face_fraction = 0.35          # face width as a fraction of a person's cell
        area = frame_w * frame_h * (face_fraction ** 2)
        return int(area // (tier_px ** 2))

    def max_students(self, frame_w=None, frame_h=None, tier_px=None):
        """
        Effective capacity: whichever ceiling is lower.

        Reporting the compute ceiling alone would promise a cohort the camera
        cannot see, and every student past the resolution ceiling would be
        counted and then reported Unknown.
        """
        return min(
            self.compute_ceiling(),
            self.resolution_ceiling(frame_w, frame_h, tier_px),
        )

    def limiting_factor(self, frame_w=None, frame_h=None, tier_px=None):
        compute = self.compute_ceiling()
        pixels = self.resolution_ceiling(frame_w, frame_h, tier_px)
        if compute <= pixels:
            return "compute"
        return "resolution"

    @property
    def admission_seconds(self):
        """
        How long a student entering the room waits to be noticed.

        Detection runs every detect_every cycles, so this grows with the
        detection interval - which is why maximising students by stretching
        that interval is not free.
        """
        return self.detect_every * self.refresh_at(self.compute_ceiling())

    def refresh_at(self, students):
        return self.cycle_ms_for(students) / 1000.0

    def as_dict(self):
        return {
            "source": self.source,
            "cores": self.cores,
            "per_face_ms": round(self.per_face_ms, 3),
            "detect_ms": round(self.detect_ms, 2),
            "pool_size": self.pool_size,
            "yolo_width": self.yolo_width,
            "detect_every": self.detect_every,
            "samples_per_gate": self.samples_per_gate,
            "shortest_gate_s": self.shortest_gate,
            "max_refresh_s": round(self.max_refresh_seconds, 3),
            "compute_ceiling": self.compute_ceiling(),
            "resolution_ceiling": self.resolution_ceiling(),
            "max_students": self.max_students(),
            "limiting_factor": self.limiting_factor(),
            "admission_s": round(self.admission_seconds, 2),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def summary(self):
        lines = [
            f"cores {self.cores}, pool {self.pool_size}, "
            f"yolo {self.yolo_width}px every {self.detect_every} cycles",
            f"per face {self.per_face_ms:.1f}ms, "
            f"detection {self.detect_ms:.0f}ms "
            f"({self.amortised_detect_ms:.0f}ms amortised)",
            f"shortest temporal gate {self.shortest_gate}s, "
            f"needs {self.samples_per_gate} samples "
            f"-> refresh must stay under {self.max_refresh_seconds:.2f}s",
        ]
        compute = self.compute_ceiling()
        pixels = self.resolution_ceiling()
        n = self.max_students()
        lines.append(
            f"compute ceiling {compute} students, "
            f"resolution ceiling {pixels} at "
            f"{CAPTURE_W_DEFAULT}x{CAPTURE_H_DEFAULT}"
        )
        if n <= 0:
            lines.append(
                "CAPACITY 0 - detection alone exceeds the fidelity budget. "
                "Lower yolo_width, raise detect_every, or relax "
                "samples_per_gate."
            )
        else:
            lines.append(
                f"CAPACITY {n} students, limited by {self.limiting_factor()} "
                f"(refresh {self.refresh_at(n):.2f}s, "
                f"a new student noticed within {self.admission_seconds:.1f}s)"
            )
        if self.source != "measured":
            lines.append(
                f"NOTE: {self.source} - run scripts/calibrate.py on this "
                f"machine for real numbers."
            )
        return "\n".join(lines)


def default_pool_size(cores=None):
    """
    Detector threads to run.

    Measured on 20 cores, the parallel speedup knee was 8 threads (3.21x, 40%
    efficiency) and 12 was past it (3.08x, 26%). The knee sits well below the
    core count because each detector is already internally threaded by XNNPACK
    and they contend. Half the cores, floor of 2, is a reasonable rule; a Pi's
    four cores give 2, which is about right and nothing like the hard-coded 8
    that was correct only for the machine this was written on.
    """
    cores = cores or os.cpu_count() or 4
    if cores >= 16:
        return 8
    return max(2, cores // 2)


def estimate_from_host(cores=None):
    """
    A rough capacity guess when nothing has been measured here.

    Deliberately crude and labelled as such. Its job is to stop the pipeline
    silently using desktop constants on a Pi, not to be accurate - the honest
    number comes from scripts/calibrate.py, run on the machine in question.
    """
    cores = cores or os.cpu_count() or 4
    pool = default_pool_size(cores)

    # Anchored on this project's own measurements: 6.7ms per detect on one
    # core of a fast x86 desktop, 2.09ms effective across a pool of 8.
    # Small-core machines are scaled by a blunt factor because per-core
    # throughput, not core count, is what dominates.
    single_core_ms = 6.7 if cores >= 16 else 25.0
    parallel_gain = min(pool, 3.2)
    per_face = single_core_ms / max(1.0, parallel_gain)

    if cores >= 16:
        yolo_width, detect_ms = 1280, 108.0
    else:
        yolo_width, detect_ms = 640, 320.0

    return Capacity(
        per_face_ms=per_face,
        detect_ms=detect_ms,
        cores=cores,
        pool_size=pool,
        yolo_width=yolo_width,
        detect_every=config.DETECT_EVERY,
        samples_per_gate=config.FIDELITY_SAMPLES_PER_GATE,
        source="ESTIMATED from core count, not measured",
    )


def load(path=None):
    """
    Read the calibration this machine wrote, or fall back to an estimate.

    Written by scripts/calibrate.py. Kept out of version control on purpose:
    it describes one machine, and copying a desktop's file onto a Pi would
    reintroduce exactly the problem this module exists to prevent.
    """
    path = path or config.TUNING_PATH
    if not os.path.exists(path):
        return estimate_from_host()

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return Capacity(
            per_face_ms=float(data["per_face_ms"]),
            detect_ms=float(data["detect_ms"]),
            cores=int(data.get("cores") or os.cpu_count() or 4),
            pool_size=int(data["pool_size"]),
            yolo_width=int(data["yolo_width"]),
            detect_every=int(data["detect_every"]),
            samples_per_gate=float(
                data.get("samples_per_gate", config.FIDELITY_SAMPLES_PER_GATE)
            ),
            source="measured",
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Could not read {path} ({exc}); falling back to an estimate.",
              flush=True)
        return estimate_from_host()


def save(capacity, path=None):
    path = path or config.TUNING_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(capacity.as_dict(), fh, indent=2)
    return path
