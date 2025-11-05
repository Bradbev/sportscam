from camerapath import CameraPath

class Highlight:
    def __init__(self, start_time):
        self.start_time = start_time
        self.end_time = start_time
        self.camera_path = CameraPath()
        self.slowmo = False

    def close(self, end_time, slowmo):
        if len(self.camera_path.camera_targets) > 0:
            self.camera_path.camera_targets[-1].time = end_time
        self.end_time = end_time
        self.slowmo = slowmo

    def get_camera_path(self):
        return self.camera_path

    def __str__(self):
        return f"Highlight from {self.start_time} to {self.end_time}"

    def __repr__(self):
        return self.__str__()


class Highlights:
    def __init__(self):
        self.highlights = []
        self.active_save_highlight = None

    def start_highlight(self, time):
        if self.active_save_highlight is not None:
            return
        self.active_save_highlight = Highlight(time)

    def stop_highlight(self, time, slowmo):
        if self.active_save_highlight is None:
            return
        self.active_save_highlight.close(time, slowmo)
        self.highlights.append(self.active_save_highlight)
        self.active_save_highlight = None

    def get_highlights(self):
        return self.highlights

    def get_active_save_highlight(self):
        return self.active_save_highlight

    def get_highlight_at_time(self, time):
        for highlight in self.highlights:
            if highlight.start_time <= time <= highlight.end_time:
                return highlight
        return None

    def get_highlight_before(self, time):
        result = None
        for highlight in self.highlights:
            if highlight.start_time <= time:
                result = highlight
            if highlight.start_time > time:
                break
        return result

    def get_highlight_after(self, time):
        for highlight in self.highlights:
            if highlight.start_time > time:
                return highlight

   

    def delete_at_time(self, time):
        for highlight in self.highlights:
            if highlight.start_time <= time <= highlight.end_time:
                self.highlights.remove(highlight)
                return highlight

    def get_highlight_by_end_time(self, time):
        for highlight in self.highlights:
            if time + 100 <= highlight.end_time < time+200:
                return highlight
        return None


    