"""CameraPath: bisect lookup, replace_range and the render start rule."""
import random

from camerapath import CameraPath
from cameratarget import CameraTarget


def path_of(times, source="human"):
    p = CameraPath()
    p.camera_targets = [CameraTarget(t, x=i, source=source) for i, t in enumerate(times)]
    return p


def linear_camera_index(targets, time):
    """The original linear scan, kept as the reference."""
    index = 0
    while index < len(targets) - 1:
        if time > targets[index + 1].time:
            index += 1
            continue
        return index
    return index


def test_camera_index_matches_the_linear_scan():
    rng = random.Random(1)
    for n in (0, 1, 2, 5, 50):
        times = sorted(rng.sample(range(0, 100000, 40), n))
        p = path_of(times)
        probes = [-5, 0] + times + [t + 1 for t in times] + [t - 1 for t in times] + [200000]
        for t in probes:
            assert p.camera_index(t) == linear_camera_index(p.camera_targets, t), (n, t)


def test_replace_range_only_touches_the_range():
    p = path_of(range(0, 10000, 1000), source="auto")
    p.replace_range(3000, 5000, [CameraTarget(3500, x=99), CameraTarget(4200, x=98)])
    assert [c.time for c in p.camera_targets] == [0, 1000, 2000, 3500, 4200, 6000, 7000, 8000, 9000]
    assert [c.source for c in p.camera_targets].count("human") == 2
    p.replace_range(10000, 20000, [])  # empty range: no change
    assert len(p.camera_targets) == 9


def test_start_is_first_target_unless_auto_and_recorded_are_mixed():
    p = path_of(range(0, 10000, 1000), source="auto")
    assert p.get_start_time() == 0
    p.replace_range(4000, 4000, [CameraTarget(4000, x=5)])  # a recorded point at the faceoff
    assert p.get_start_time() == 4000
    old = path_of([2000, 3000])  # pickles from before `source` existed count as recorded
    for c in old.camera_targets:
        del c.source
    assert old.get_start_time() == 2000
    assert CameraPath().get_start_time() == 0
