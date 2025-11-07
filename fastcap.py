from os import path
import queue
import threading
import time
import cv2


class FastCap():
    def __init__(self, filename):
        self.filename = filename
        self.cap = cv2.VideoCapture(filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = queue.Queue(maxsize=30)
        self.last_time = 0
        self.running = True

    def read_frame(self):
        success, frame = self.cap.read()
        self.last_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.running = success
        return success, frame

    def set_time(self, time):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, time)

    def get_time(self):
        self.last_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return self.last_time

    def get_fps(self):
        return self.fps

    def get_frame_count(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    def isOpened(self):
        return self.running

    def release(self):
        self.running = False
        self.cap.release()
