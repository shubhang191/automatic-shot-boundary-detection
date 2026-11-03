"""Zero-based inclusive shots. Native transition intervals remain explicit."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class CandidateCut:
    a: int
    b: int
    kind: str = "unknown"
    score: float | None = None


def canonical_cuts(cuts, count):
    if not isinstance(count, int) or count < 1:
        raise ValueError("A video must contain at least one decoded frame")
    unique = {}
    for c in cuts:
        if type(c.a) is not int or type(c.b) is not int or not 0 <= c.a < c.b < count:
            raise ValueError(f"Invalid cut ({c.a},{c.b}) for {count} frames; expected 0 <= a < b < T")
        if c.score is not None and not math.isfinite(c.score):
            raise ValueError("Non-finite detector score")
        unique.setdefault((c.a, c.b), c)
    out = sorted(unique.values(), key=lambda c: (c.a, c.b))
    if any(x.b > y.a for x, y in zip(out, out[1:])):
        raise ValueError("Overlapping transition intervals cannot define an unambiguous shot sequence")
    return out


def shots_from_cuts(cuts, count, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    cuts = canonical_cuts(cuts, count)
    starts = [0] + [c.b for c in cuts]
    ends = [c.b - 1 for c in cuts] + [count - 1]
    # Transition frames a+1..b-1 belong to the preceding display/export shot.
    # They are still represented explicitly in CandidateCut and native ranges.
    return [dict(shot=i + 1, start_frame=s, end_frame=e, start_s=s / fps,
                 end_inclusive_s=e / fps, end_exclusive_s=(e + 1) / fps,
                 duration_s=(e - s + 1) / fps,
                 boundary_type="start" if i == 0 else cuts[i - 1].kind)
            for i, (s, e) in enumerate(zip(starts, ends))]


def cuts_from_native_scenes(scenes, count, kinds=None):
    last_end = -1
    for s, e in scenes:
        if not 0 <= s <= e < count or s <= last_end:
            raise ValueError("Invalid/overlapping native inclusive scenes")
        last_end = e
    return canonical_cuts([CandidateCut(int(a[1]), int(b[0]), kinds[i] if kinds else "unknown")
                           for i, (a, b) in enumerate(zip(scenes, scenes[1:]))], count)


def cut_dicts(cuts):
    return [asdict(c) for c in cuts]
