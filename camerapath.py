
from cameratarget import CameraTarget


def sign(x):
    if x < 0:
        return -1
    return 1

def cubic(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    if x < 0.5:
        return 4 * x * x * x
    return 1 - pow(-2 * x + 2, 3) / 2


class CameraPath:
    def __init__(self):
        self.camera_targets = []

    def truncate_path_to_time(self, time):
        index = self.camera_index(time)
        if index < len(self.camera_targets)-1:
            self.camera_targets = self.camera_targets[0:index]

    def add_camera_target(self, camera_target):
        self.truncate_path_to_time(camera_target.time)
        self.camera_targets.append(camera_target)

    def camera_index(self, time):
        index = 0
        while index < len(self.camera_targets)-1:
            if time > self.camera_targets[index+1].time:
                index = index + 1
                continue
            return index
        return index

    # returns x value, the current camera, and the next camera
    def get_camera_at_time(self, frame_time):
        if len(self.camera_targets) == 0:
            return 0, CameraTarget(0,0), CameraTarget(0,0)
        index = self.camera_index(frame_time)
        if index == -1:
            return 0, CameraTarget(0,0), CameraTarget(0,0)
        if index >= len(self.camera_targets)-1:
            return self.camera_targets[-1].x, self.camera_targets[-1], self.camera_targets[-1]

        targets = self.camera_targets

        current = targets[index]
        next = targets[index+1]
        fraction = (frame_time - current.time) / (next.time - current.time)

        # ease the camera if changing direction
        camera_direction = sign(next.x - current.x)
        if fraction < 0.5:
            # ease out
            if index >= 1:
                prev_dir = sign(current.x - targets[index-1].x)
                if camera_direction != prev_dir:
                    fraction = cubic(fraction)
        else:
            # ease in
            if index < len(targets)-2:
                next_dir = sign(targets[index+2].x - next.x)
                if camera_direction != next_dir:
                    fraction = cubic(fraction)

        return current.x + ((next.x - current.x) * (fraction)), current, next

    def has_targets(self):
        return len(self.camera_targets) > 0

    def get_last_cam_time(self):
        if len(self.camera_targets) == 0:
            return 0
        return self.camera_targets[-1].time

    def get_next_cam_time(self, time):
        index = self.camera_index(time)
        if index == -1:
            return 0
        if index < len(self.camera_targets)-1:
            return self.camera_targets[index+1].time
        return self.camera_targets[index].time
    
    def get_prev_cam_time(self, time):
        index = self.camera_index(time)
        if index == -1:
            return 0
        if index > 0:
            return self.camera_targets[index-1].time
        return self.camera_targets[index].time

    def to_string(self, frame_time):
        index = self.camera_index(frame_time)+1
        return f"{index}/{len(self.camera_targets)}"

    def __repr__(self):
        return str(self.camera_targets)
