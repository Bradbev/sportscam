import glob
import math
from pathlib import Path
import time
import numpy as np
import cv2
import pickle
import humanize
from datetime import datetime, timedelta
from os import path
import argparse
from camerapath import CameraPath
from cameratarget import CameraTarget
import fastcap
from highlight import Highlights

parser = argparse.ArgumentParser()
parser.add_argument("basepath", help="Base path to begin video processing from")
parser.add_argument("-r", "--render", action="store_true", help="Render the video")
parser.add_argument("-ns", "--no-skip", action="store_true", help="Don't skip files that are finished")
parser.add_argument("-i", "--iso", action="store_true", help="When ISO is active, skip forward to the next camera shot if the gap is > 30s")
parser.add_argument("--logo", type=str, help="Logo screen wipe")
parser.add_argument("--show-logo", action="store_true", help="Only show logo")
args = parser.parse_args()

writeOutputFile = args.render
basePath = args.basepath
preview = True
iso = args.iso
angleCam = True
logo_frames = []
logo_masks = []

def scale(sz, s=0.25):
    return (int(sz[0]*s),int(sz[1]*s))

mouse_x = 0
mouse_click = False

inputSize = (3840, 2880)
viewSize = (3340, 1050)
outSize = (1920,720)
fullOutSize = (1920, 1080)
mini_view_size = (400, 400)
mini_view_rot_size = (300, 180)

diag = int(math.ceil(math.sqrt(outSize[0]**2 + outSize[1]**2))) + 100
roiCapture = (diag,diag)
if not angleCam:
    roiCapture = outSize

def ms_str(ms):
    return humanize.precisedelta(timedelta(milliseconds=ms))

def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

def rotate_image_crop(img, angle):
    rows, cols, _ = img.shape
    center = (cols // 2, rows // 2)
    scale = 1.0 # No scaling
    # Get the 2D rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (cols, rows))

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_click
    mouse_x = int((x-outSize[0]/2)*2)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click = True

def load_pkl(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except:
        pass

def save_pkl(filename, data):
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    except:
        pass

def add_logo_to_frame(frame, logo_index):
    if len(logo_frames) == 0:
        return frame
    logo = logo_frames[logo_index]
    inverted_mask = logo_masks[logo_index]
    mask = 255-inverted_mask
    maskedLogo = cv2.bitwise_and(logo, logo, mask=mask)
    maskedBack = cv2.bitwise_and(frame, frame, mask=inverted_mask)
    return cv2.add(maskedBack, maskedLogo)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback)

class Processor:
    def __init__(self, filename):
        print(filename)
        self.filename = filename
        self.paused = False
        self.auto_record = False
        self.last_auto_time = 0
        self.mini_view_x = 3550
        self.mini_view_y = 2000
        self.rotation = 25
        self.mini_rotation = 8
        self.top_of_roi = 334
        self.had_pickle = False
        self.angle_left = -9
        self.angle_right = 9
        self.pureFrame = None
        self.slowmo = False
        self.play_highlights = True #writeOutputFile
        self.pending_active_playback_highlight = None
        self.active_playback_highlight = None
        loaded = load_pkl(filename + '.pkl')
        self.camera_path = CameraPath()
        self.highlights = Highlights()
        if loaded is not None:
            self.had_pickle = True
            self.top_of_roi = loaded["top_of_roi"]
            self.rotation = loaded["rotation"]
            self.mini_rotation = loaded["mini_rotation"]
            self.angle_left = loaded["angle_left"]
            self.angle_right = loaded["angle_right"]
            self.camera_path = loaded["camera_path"]
            if "mini_view_x" in loaded:
                self.mini_view_x = loaded["mini_view_x"]
            if "mini_view_y" in loaded:
                self.mini_view_y = loaded["mini_view_y"]
            if "highlights" in loaded:
                self.highlights = loaded["highlights"]
                self.highlights.active_save_highlight = None


    def set_last_pickle(self, pkl):
        if not pkl:
            return
        if not self.had_pickle:
            # load some data from the previous pickle
            self.top_of_roi = pkl["top_of_roi"]
            self.rotation = pkl["rotation"]
            if "mini_view_x" in pkl:
                self.mini_view_x = pkl["mini_view_x"]
            if "mini_view_y" in pkl:
                self.mini_view_y = pkl["mini_view_y"]
            cam_path = pkl["camera_path"]
            self.camera_path = CameraPath()
            self.camera_path.add_camera_target(CameraTarget(0, cam_path.camera_targets[-1].time))
            if "auto_record" in pkl:
                self.auto_record = pkl["auto_record"]

    def set_time(self, time):
        self.auto_record = False
        self.cap.set_time(time)
        self.last_auto_time = 0

    def skip_time(self, delta):
        time = self.cap.get_time()
        frame_time = time+delta
        if frame_time < 0:
            frame_time = 0
        self.set_time(frame_time)

    def isRunning(self):
        return self.cap.isOpened()

    def isPaused(self):
        return self.paused

    def handleKeys(self, key, frame_time):
        # Pausing
        if key == ord(' '):
           self.paused = not self.paused

        # highlights
        if key == ord('h'):
            if self.highlights.get_active_save_highlight() is not None:
                self.highlights.stop_highlight(frame_time, self.slowmo)
                self.auto_record = False
            else:
                self.auto_record = True
                self.highlights.start_highlight(frame_time)

        if key == ord('g'):
            self.play_highlights = not self.play_highlights

        if key == ord('d'):
            self.highlights.delete_at_time(frame_time)

        if key == ord('s'):
            self.slowmo = not self.slowmo

        # Rotation and height
        if key == ord('U'):
            self.mini_rotation -= 0.2
        if key == ord('I'):
            self.mini_rotation += 0.2

        if key == ord('u'):
            if mouse_x < 500:
                self.angle_left = self.angle_left-0.2
            elif mouse_x > 3000:
                self.angle_right = self.angle_right-0.2
            else:
                self.rotation = self.rotation-0.2
        if key == ord('i'):
            if mouse_x < 500:
                self.angle_left = self.angle_left+0.2
            elif mouse_x > 3000:
                self.angle_right = self.angle_right+0.2
            else:
                self.rotation = self.rotation+0.2
        if key == ord('j'):
            self.top_of_roi = self.top_of_roi-1
        if key == ord('k'):
            self.top_of_roi = self.top_of_roi+1            
        
        # Mini-view position
        if key == ord(','): # <
            self.mini_view_x -= 1
        if key == ord('.'): # >
            self.mini_view_x += 1
        if key == ord('n'):
            self.mini_view_y -= 1
        if key == ord('m'):
            self.mini_view_y += 1

        # Time and recording controls
        if key == ord('a'):
            self.auto_record = not self.auto_record
        if key == ord('1'):
            self.skip_time(-1000 * 15)
        if key == ord('2'):
            self.skip_time(-1000 * 5)
        if key == ord('3'):
            self.skip_time(1000 * 10)
        if key == ord('4'):
            self.skip_time(1000 * 60)

        # Camera controls
        if self.camera_path.has_targets():
            if key == ord('0'):
                self.set_time(self.camera_path.get_last_cam_time())
            if key == ord('9'):
                self.set_time(self.camera_path.get_next_cam_time(frame_time))
            if key == ord('8'):
                self.set_time(self.camera_path.get_prev_cam_time(frame_time))
            if key == ord('7'):
                self.set_time(0)


    def readFrame(self):
        return self.cap.read_frame()

    def _create_output_frame(self, source_frame, x_pos, max_x):
        """Helper function to create the final output frame (cropped, rotated, letterboxed)."""
        (w,h) = roiCapture
        y = self.top_of_roi * 5
        outFrame = source_frame[y:y+h, x_pos:x_pos+w]

        if angleCam:
            angle = np.interp([x_pos], [0, max_x], [self.angle_left, self.angle_right])[0]
            y_adjust = int(np.interp([x_pos], [0, max_x/2, max_x], [200,0,200])[0])
            outFrame = rotate_image(outFrame, angle)
            xr, yr = int(diag/2-outSize[0]/2), int(diag/2-outSize[1]/2) + y_adjust - 400
            outFrame = outFrame[yr:yr+outSize[1], xr:xr+outSize[0]]

        # Add black bars to make it full HD
        top_border = (fullOutSize[1] - outSize[1]) // 2
        bottom_border = fullOutSize[1] - outSize[1] - top_border
        outFrame = cv2.copyMakeBorder(outFrame, top_border, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        return outFrame

    def process(self):
        global mouse_x
        global mouse_click
        camera_x = 0
        self.cap = fastcap.FastCap(self.filename+'.MP4')

        fps = self.cap.get_fps()
        frame_ms = 1000 / fps
        print(f"FPS: {fps}, FrameMS {frame_ms}")

        if writeOutputFile:
            out = cv2.VideoWriter(path.join(basePath, self.filename+"_processed") + '.mp4', cv2.VideoWriter_fourcc(*"acv1"), fps, fullOutSize)
            print(f"Rendering to {path.join(basePath, self.filename+"_processed") + '.mp4'} {fullOutSize}")

        # Loop through the video frames
        start_time = datetime.now()
        next_print = datetime.now() + timedelta(seconds=5)
        frame_count = 0
        frame = None
        success = False
        wipe_index = 100000
        on_wipe_end = None
        is_wiping = False
        def do_wipe(on_done):
            nonlocal wipe_index
            nonlocal on_wipe_end
            nonlocal is_wiping
            on_wipe_end = on_done
            wipe_index = 0
            is_wiping = True

        while self.isRunning():
            frame_count = frame_count + 1
            is_wiping = wipe_index < len(logo_frames)

            if self.isPaused():
                frame = self.pureFrame
                success = True
            elif self.slowmo and frame_count % 3 != 0:
                frame = self.pureFrame
                success = True
            else:
                success, frame = self.readFrame()

            if success:
                self.pureFrame = frame
                # Capture a 200x200 rectangle from a fixed location in the original frame.
                # You can adjust the coordinates (e.g., 100, 100) as needed.
                mini_view = frame[self.mini_view_y:self.mini_view_y+mini_view_size[1], self.mini_view_x:self.mini_view_x+mini_view_size[0]]
                mini_view = rotate_image(mini_view, self.rotation+self.mini_rotation)
                dx,dy=75,130
                mini_view = mini_view[dy:dy+mini_view_rot_size[1],dx:dx+mini_view_rot_size[0]]
                
                frame_time = self.cap.get_time()
                frame = rotate_image(frame, self.rotation)

                # get the normal camera track
                camera_x, next_cam_point, cam_index = self.camera_path.get_camera_at_time(frame_time)

                # handle iso selection by advancing to the next camera if the gap is > 30s
                # do the same if it's the start of the game
                if (iso or cam_index == 0) and next_cam_point.time - frame_time > 30000:
                    self.set_time(next_cam_point.time)
                    continue

                saving_highlight = self.highlights.get_active_save_highlight()
                if saving_highlight is not None:
                    camera_x, next_cam_point, cam_index = saving_highlight.get_camera_path().get_camera_at_time(frame_time)
                
                if not saving_highlight and self.play_highlights and not self.pending_active_playback_highlight:
                    h = self.highlights.get_highlight_by_end_time(frame_time)
                    if h is not None:
                        self.pending_active_playback_highlight = h
                        def start_highlight():
                            h = self.pending_active_playback_highlight
                            self.active_playback_highlight = h
                            self.slowmo = h.slowmo
                            self.set_time(h.start_time)
                        do_wipe(start_highlight)
                            
                #if not is_wiping and self.pending_active_playback_highlight is not None and self.active_playback_highlight is None:
                    #h = self.pending_active_playback_highlight
                    #self.active_playback_highlight = h
                    #self.slowmo = h.slowmo
                    #self.set_time(h.start_time)
                    #continue

                if not is_wiping and self.active_playback_highlight and frame_time > self.active_playback_highlight.end_time-2000:
                    time_to_set = self.active_playback_highlight.end_time+400
                    def end_highlight():
                        self.set_time(time_to_set)
                        self.active_playback_highlight = None
                        self.pending_active_playback_highlight = None
                        self.slowmo = False
                    do_wipe(end_highlight)
                
                if self.active_playback_highlight:
                    camera_x, next_cam_point, cam_index = self.active_playback_highlight.get_camera_path().get_camera_at_time(frame_time)

                max_x = frame.shape[1]-outSize[0]
                x = int(np.clip(camera_x, 0, max_x))
                clamped_mouse_x = int(np.clip(mouse_x, 0, frame.shape[1]-outSize[0]))

                if mouse_click or (self.auto_record and frame_time > self.last_auto_time + 1000):
                    if self.auto_record:
                        self.last_auto_time = frame_time

                    if saving_highlight is not None:
                        saving_highlight.get_camera_path().add_camera_target(CameraTarget(frame_time, clamped_mouse_x))
                    else:
                        self.camera_path.add_camera_target(CameraTarget(frame_time, clamped_mouse_x))
                    mouse_click = False

                use_live_cam = self.auto_record or self.isPaused() or frame_time > next_cam_point.time
                frame_x = clamped_mouse_x if use_live_cam else x

                outFrame = self._create_output_frame(frame, frame_x, max_x)
                # Composite the captured mini-view onto the output frame at (0,0)
                outFrame[0:mini_view_rot_size[1], 0:mini_view_rot_size[0]] = mini_view

                if is_wiping:
                    outFrame = add_logo_to_frame(outFrame, wipe_index)
                    wipe_index = wipe_index + 1
                    if wipe_index >= len(logo_frames) and on_wipe_end:
                        on_wipe_end()


                if writeOutputFile:
                    if datetime.now() > next_print:
                        delta = datetime.now() - start_time
                        print(f"Processed {frame_count} frames in {delta} ({frame_count / delta.seconds} fps)")
                        next_print = datetime.now() + timedelta(seconds=30)

                    out.write(outFrame)

                if preview:
                    cv2.imshow('preview',outFrame)
                
                if not writeOutputFile:
                    # Display the annotated main frame
                    text_y = 1550
                    line1 = f"{self.camera_path.to_string(frame_time)} @ {ms_str(next_cam_point.time)} {len(self.highlights.get_highlights())} highlights"
                    if saving_highlight is not None:
                        line1 += f" Saving Highlight {saving_highlight.get_camera_path().to_string(frame_time)}"
                    cv2.putText(frame, line1, (0,text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4)

                    line2 = f"{ms_str(frame_time)} {"Auto" if self.auto_record else ""}"
                    if self.play_highlights:
                        line2 += " will play highlights"
                    if self.active_playback_highlight:
                        line2 += " highlight active"
                    cv2.putText(frame, line2, (0,text_y+100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4)

                    (w,h) = outSize # This is used for the rectangle drawing below
                    y = self.top_of_roi*5
                    if angleCam:
                        y = y + int(720/2)

                    if (not self.auto_record) and next_cam_point.time >= frame_time:
                        blue = (255,0,0)
                        cv2.rectangle(frame, (x,y),(x+w,y+h), blue, 10)

                    if self.active_playback_highlight is not None:
                        hx,_,_ = self.active_playback_highlight.get_camera_path().get_camera_at_time(frame_time)
                        hx = int(np.clip(hx, 0, max_x))
                        cyan = (255,255,0)
                        cv2.rectangle(frame, (hx,y),(hx+w,y+h), cyan, 5)

                    purple = (255,0,200)
                    red = (55,0,200)
                    rect_col = purple if self.auto_record else red
                    cv2.rectangle(frame, (clamped_mouse_x,y),(clamped_mouse_x+w,y+h), rect_col, 10)

                    frame = frame[1500:3200,0:frame.shape[1]]
                    frame = cv2.resize(frame, viewSize)

                    cv2.imshow('frame',frame)


                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                self.handleKeys(key, frame_time)
            else:
                # Break the loop if the end of the video is reached
                break
        finished = not self.isRunning()
        toSave = {
            "top_of_roi" : self.top_of_roi,
            "rotation" : self.rotation,
            "camera_path" : self.camera_path,
            "finished" : finished,
            "rendered" : writeOutputFile and finished,
            "auto_record":self.auto_record,
            "mini_view_x": self.mini_view_x, 
            "mini_view_y": self.mini_view_y,
            "highlights":self.highlights,
            "angle_left":self.angle_left,
            "angle_right":self.angle_right,
            "mini_rotation":self.mini_rotation
        }
        save_pkl(self.filename+'.pkl', toSave)

        # Release the video capture object and close the display window
        self.cap.release()
        if writeOutputFile:
            out.release()

        return finished

def load_logo():
    global logo_frames
    global logo_masks
    cap = fastcap.FastCap(args.logo)
    while True:
        success, frame = cap.read_frame()
        if not success:
            break
        frame = cv2.resize(frame, fullOutSize)
        logo_frames.append(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([153,15,85])
        upper = np.array([170,255,255])
        inverted_mask = cv2.inRange(hsv, lower, upper)
        logo_masks.append(inverted_mask)
    cap.release
 
def show_logo():
    cap2 = fastcap.FastCap("d:\\RawVideo\\panthres\\DJI_20251014181117_0009_D.MP4")
    
    for i,logo in enumerate(logo_frames):
        inverted_mask=logo_masks[i]
        mask = 255-inverted_mask

        _, frame = cap2.read_frame()
        frame = cv2.resize(frame,(1920,1080))
        bf = add_logo_to_frame(frame, i)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        #cv2.imshow('frame', frame)
        #cv2.imshow('mask', inverted_mask)
        #cv2.imshow('maskLogo', maskedLogo)
        #cv2.imshow('preview', maskedBack)
#        cv2.imshow('hock', f2)
        cv2.imshow('final', bf)

    cap2.release()

def examine_files():
    raw_files = glob.glob(path.join(basePath, "*.mp4"))
    sources = [Path(f) for f in raw_files if not "_processed" in f]
    last_pickle = None
    state = None
    for fp in sources:
        state = load_pkl(fp.with_suffix(".pkl"))
        process_file = True
        if state and not args.no_skip:
            process_file = state["finished"] is False
            if writeOutputFile:
                process_file = not state["rendered"]

        if process_file:
            p = Processor(str(fp.with_suffix("")))
            p.set_last_pickle(last_pickle)
            print(f"Starting process for {fp}")
            if not p.process():
                break
            print(f"process for {fp} done")
            # reload the pickle that the process() call saved
            last_pickle = load_pkl(fp.with_suffix(".pkl"))

load_logo()
if args.show_logo:
    show_logo()
else:
    examine_files()
cv2.destroyAllWindows()
