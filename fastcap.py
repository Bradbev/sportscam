from os import path
import queue
import threading
import time
import cv2


class FastCap():
    def __init__(self, filenames):
        self.filenames = filenames
        self.caps = []
        self.total_frame_count = 0
        for f in filenames:
            cap = cv2.VideoCapture(f)
            self.caps.append(cap)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.total_frame_count += count

        self.active_cap_index = 0

        self._set_active_cap(0)

        self.fps = self.active_cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1000 / self.fps

        self.cap_time_bases = []
        offset = 0
        for cap in self.caps:
            self.cap_time_bases.append(offset)
            offset += cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.frame_time
        self.cap_time_bases.append(offset)

        self.running = True

    def _set_active_cap(self, index):
        self.active_cap_index = index
        self.active_cap = self.caps[index]

    def read_frame(self):
        success, frame = self.active_cap.read()
        if not success:
            cap_index = self.caps.index(self.active_cap)
            if cap_index < len(self.caps)-1:
                self._set_active_cap(cap_index+1)
                self.active_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                success, frame = self.active_cap.read()

        self.running = success
        return success, frame

    def set_time(self, time):
        for i, cap in enumerate(self.caps):
            if time < self.cap_time_bases[i+1]:
                self._set_active_cap(i)
                cap_time = time - self.cap_time_bases[i]
                cap.set(cv2.CAP_PROP_POS_MSEC, cap_time)
                return

    def get_time(self):
        return self.cap_time_bases[self.active_cap_index] + self.active_cap.get(cv2.CAP_PROP_POS_MSEC)

    def get_cap_index(self):
        return self.active_cap_index

    def get_fps(self):
        return self.fps

    def get_frame_count(self):
        return self.total_frame_count
    
    def isOpened(self):
        return self.running

    def release(self):
        self.running = False
        self.active_cap.release()
