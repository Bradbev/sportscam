from os import path
import queue
import threading
import time
import cv2


def decode_thread(fastcap):
    print("Decode thread")
    while fastcap.running:
        with fastcap.lock:
            if not fastcap.running:
                break
            success, frame = fastcap.cap.read()
            frame_time = fastcap.cap.get(cv2.CAP_PROP_POS_MSEC)
            fastcap.frames.put((success, frame, frame_time))
            if not success:
                fastcap.running = False
                break

class FastCap():
    def __init__(self, filename):
        self.filename = filename
        self.cap = cv2.VideoCapture(filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = queue.Queue(maxsize=30)
        self.last_time = 0
        self.lock = threading.Lock()
        self.running = True
        #t = threading.Thread(target=decode_thread, args=(self, ), daemon=True)
        #t.start()

    def read_frame(self):
        #(success, frame, frame_time) = self.frames.get()
        #self.last_time = frame_time
        success, frame = self.cap.read()
        self.last_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.running = success
        return success, frame

    def set_time(self, time):
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time)
            #while not self.frames.empty():
                #self.frames.get()

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
        with self.lock:
            self.running = False
        self.cap.release()
