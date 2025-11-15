import cv2
import numpy as np
import fastcap


class LogoPlayer:
	def __init__(self, path, full_out_size):
		self.frames = []
		self.masks = []
		self.is_playing = False
		self.index = 0
		self.on_wipe_end = None

		cap = fastcap.FastCap([path])
		while True:
			success, frame = cap.read_frame()
			if not success:
				break
			frame = cv2.resize(frame, full_out_size)
			self.frames.append(frame)
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			lower = np.array([153,15,85])
			upper = np.array([170,255,255])
			inverted_mask = cv2.inRange(hsv, lower, upper)
			self.masks.append(inverted_mask)
		cap.release

	def add_logo_if_needed(self, frame):
		if not self.is_playing or len(self.frames) == 0:
			return frame
		logo = self.frames[self.index]
		inverted_mask = self.masks[self.index]
		self.index += 1

		if self.index >= len(self.frames) and self.on_wipe_end:
			self.on_wipe_end()
			self.is_playing = False
			self.index = 0

		mask = 255 - inverted_mask
		maskedLogo = cv2.bitwise_and(logo, logo, mask=mask)
		maskedBack = cv2.bitwise_and(frame, frame, mask=inverted_mask)
		return cv2.add(maskedBack, maskedLogo)

	def do_wipe_then(self, on_wipe_end):
		self.on_wipe_end = on_wipe_end
		self.is_playing = True