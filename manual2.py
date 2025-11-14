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
from util import rotate_point

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
logo_frames = []
logo_masks = []

def scale(sz, s=0.25):
    return (int(sz[0]*s),int(sz[1]*s))

mouse_x = 0
mouse_y = 0
mouse_click = False

view_size = (3340, 1050)
out_size = (1920,720)
full_out_size = (1920, 1080)
mini_view_size = (400, 400)
mini_view_rot_size = (350, 180)

roi_size = out_size

def ms_str(ms):
    return str(timedelta(milliseconds=ms)).split(".")[0]

def rotate_image(img, angle, scale=1.0):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, scale)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

def rotate_image_crop(img, angle, scale=1.0):
    rows, cols, _ = img.shape
    center = (cols // 2, rows // 2)
    # Get the 2D rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (cols, rows))

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_click
    mouse_x = int((x-out_size[0]/2)*2)
    mouse_y = int((y-out_size[1]/2)*2)
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
    def __init__(self, filenames):
        print(filenames)
        self.filenames = filenames
        self.filename = filenames[0]
        self.paused = False
        self.auto_record = False
        self.last_auto_time = 0
        self.mini_view_x = 3550
        self.mini_view_y = 2000
        self.rotation = 37
        self.mini_rotation = 8
        self.top_of_roi = -1600
        self.top_of_roi_left = 0
        self.top_of_roi_right = 0
        self.had_pickle = False
        self.angle_left = -9
        self.angle_right = 9
        self.center_x_offset = 0
        self.last_raw_frame = None
        self.slowmo = False
        self.zoom = 1.0
        self.play_highlights = writeOutputFile
        self.do_camera_cuts = writeOutputFile
        self.finished_rendering = False
        self.pending_active_playback_highlight = None
        self.active_playback_highlight = None
        loaded = load_pkl(self.filename + '.pkl')
        self.camera_path = CameraPath()
        self.highlights = Highlights()
        if loaded is not None:
            self.had_pickle = True
            self.load_camera_config_pickle(loaded)

            self.camera_path = loaded["camera_path"]
            self.highlights = loaded["highlights"]
            self.highlights.active_save_highlight = None

    def load_camera_config_pickle(self, pkl):
        self.top_of_roi = pkl["top_of_roi"]
        self.top_of_roi_left = pkl["top_of_roi_left"]
        self.top_of_roi_right = pkl["top_of_roi_right"] 
        self.rotation = pkl["rotation"]
        self.mini_rotation = pkl["mini_rotation"]
        self.angle_left = pkl["angle_left"]
        self.angle_right = pkl["angle_right"]
        self.center_x_offset = pkl["center_x_offset"]
        self.mini_view_x = pkl["mini_view_x"]
        self.mini_view_y = pkl["mini_view_y"]

    def save_pickle(self, finished):
        toSave = {
            # Camera config
            "top_of_roi" : self.top_of_roi,
            "top_of_roi_left" : self.top_of_roi_left,  
            "top_of_roi_right" : self.top_of_roi_right,
            "rotation" : self.rotation,
            "mini_rotation":self.mini_rotation,
            "angle_left":self.angle_left,
            "angle_right":self.angle_right,
            "mini_view_x": self.mini_view_x, 
            "mini_view_y": self.mini_view_y,
            "center_x_offset":self.center_x_offset,
            ### end Camera config
            
            ### end Camera config
            # per file state
            "camera_path" : self.camera_path,
            "auto_record":self.auto_record,
            "highlights":self.highlights,
            # processing state
            "rendered" : writeOutputFile and finished,
            "finished" : finished,
        }
        save_pkl(self.filename+'.pkl', toSave)


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
        return self.cap.isOpened() and not self.finished_rendering

    def isPaused(self):
        return self.paused

    def handleKeys(self, key, frame_time):
        #zoom 
        if key == ord('z'):
            self.zoom = min(3.0, self.zoom + 0.1)
        if key == ord('Z'):
            self.zoom = max(1.0, self.zoom - 0.1)

        # Hard camera cut
        highlight_is_saving = self.highlights.get_active_save_highlight() is not None
        if key == ord('c'):
            if not highlight_is_saving and len(self.camera_path.camera_targets) > 0:
                self.camera_path.camera_targets[-1].cut_to = True
                self.auto_record = False

        # Pausing
        if key == ord(' '):
           self.paused = not self.paused

        # highlights
        if key == ord('h'):
            if highlight_is_saving:
                # ending a highlight also trims the regular camera path so that cuts will work, and auto marks it as a cut
                if len(self.highlights.get_active_save_highlight().get_camera_path().camera_targets) > 0:
                    ct = self.highlights.get_active_save_highlight().get_camera_path().camera_targets[-1]
                    self.camera_path.truncate_path_to_time(frame_time)
                    self.camera_path.add_camera_target(CameraTarget(frame_time+1000, ct.x, ct.y, cut_to=True, zoom=1.0))
                    self.highlights.stop_highlight(frame_time, self.slowmo)
                else:
                    self.highlights.abort_highlight()
                self.auto_record = False
                self.zoom = 1.0
            else:
                self.highlights.start_highlight(frame_time)
        if key == ord('@'):
            h = self.highlights.get_highlight_before(frame_time)
            if h is not None:
                self.set_time(h.start_time)
        if key == ord('#'):
            h = self.highlights.get_highlight_after(frame_time)
            if h is not None:
                self.set_time(h.start_time)


        if key == ord('g'):
            self.play_highlights = not self.play_highlights
            self.do_camera_cuts = not self.do_camera_cuts

        if key == ord('d'):
            self.highlights.delete_at_time(frame_time)

        if key == ord('s'):
            self.slowmo = not self.slowmo

        # Rotation and height
        if key == ord('U'):
            self.mini_rotation -= 0.2
        if key == ord('I'):
            self.mini_rotation += 0.2

        if key == ord('x'):
            self.center_x_offset -= 1
        if key == ord('X'):
            self.center_x_offset += 1

        change_left = mouse_x < 500
        change_right = mouse_x > 3000

        if key == ord('u'):
            if change_left:
                self.angle_left = self.angle_left-0.2
            elif change_right:
                self.angle_right = self.angle_right-0.2
            else:
                self.rotation = self.rotation-0.2
        if key == ord('i'):
            if change_left:
                self.angle_left = self.angle_left+0.2
            elif change_right:
                self.angle_right = self.angle_right+0.2
            else:
                self.rotation = self.rotation+0.2

        if key == ord('j'):
            if change_left:
                self.top_of_roi_left -= 1
            elif change_right:
                self.top_of_roi_right -= 1
            else:
                self.top_of_roi = self.top_of_roi-5
        if key == ord('k'):
            if change_left:
                self.top_of_roi_left += 1
            elif change_right:
                self.top_of_roi_right += 1
            else:
                self.top_of_roi = self.top_of_roi+5
        
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
        if key == ord('`'):
            self.skip_time(-1000 * 60)
        if key == ord('1'):
            self.skip_time(-1000 * 15)
        if key == ord('2'):
            self.skip_time(-1000 * 5)
        if key == ord('3'):
            self.skip_time(1000 * 9)
        if key == ord('4'):
            self.skip_time(1000 * 60)

        # Camera controls
        if self.camera_path.has_targets():
            if key == ord('8'):
                self.set_time(self.camera_path.get_prev_cam_time(frame_time))
            if key == ord('9'):
                self.set_time(self.camera_path.get_next_cam_time(frame_time))
            if key == ord('0'):
                self.set_time(self.camera_path.get_last_cam_time())


    def readFrame(self):
        return self.cap.read_frame()

    def capture_mini_view(self, raw_frame):
        # Capture a 200x200 rectangle from a fixed location in the original frame.
        mini_view = raw_frame[self.mini_view_y:self.mini_view_y+mini_view_size[1], self.mini_view_x:self.mini_view_x+mini_view_size[0]]
        mini_view = rotate_image(mini_view, self.rotation+self.mini_rotation)
        dx,dy=75,130
        return mini_view[dy:dy+mini_view_rot_size[1],dx:dx+mini_view_rot_size[0]]
 

    def _create_output_frame(self, source_frame, x_pos, y_pos, max_x, zoom, show_target):
        """Helper function to create the final output frame (cropped, rotated, letterboxed)."""
        (w,h) = (int(roi_size[0] / zoom), int(roi_size[1] / zoom))
        x_pos = int(min(x_pos, source_frame.shape[1]-w))

        y = y_pos

        roi_frame = source_frame[y:y+h, x_pos:x_pos+w]
        if zoom > 1.0:
            roi_frame = cv2.resize(roi_frame, out_size)

        if show_target:
            red = (0,0,255)
            rx, ry = int(out_size[0]/2), int(out_size[1]/2)
            y_off = int(out_size[1]/6)
            cv2.line(roi_frame, (rx-50,ry),(rx+50,ry), red, 1)
            cv2.line(roi_frame, (rx,ry-y_off),(rx,ry+y_off), red, 1)
            cv2.line(roi_frame, (rx-50,ry-y_off),(rx+50,ry-y_off), red, 1)
            cv2.line(roi_frame, (rx-50,ry+y_off),(rx+50,ry+y_off), red, 1)

        # Add black bars to make it full HD
        top_border = (full_out_size[1] - out_size[1]) // 2
        bottom_border = full_out_size[1] - out_size[1] - top_border
        result = cv2.copyMakeBorder(roi_frame, top_border, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        #cv2.line(result, (result.shape[1]//2,0),(result.shape[1]//2,result.shape[0]), (255,255,255), 1)
        return result

    def process(self):
        global mouse_x
        global mouse_click
        self.cap = fastcap.FastCap([x + ".mp4" for x in self.filenames])

        fps = self.cap.get_fps()
        frame_ms = 1000 / fps
        total_vid_time = self.cap.get_frame_count()*frame_ms
        print(f"FPS: {fps}, FrameMS {frame_ms}, Total Video Time: {total_vid_time}")

        if writeOutputFile:
            out = cv2.VideoWriter(path.join(basePath, self.filename+"_processed") + '.mp4', cv2.VideoWriter_fourcc(*"acv1"), fps, full_out_size)
            print(f"Rendering to {path.join(basePath, self.filename+"_processed") + '.mp4'} {full_out_size}")

        # Loop through the video frames
        start_time = datetime.now()
        next_print = datetime.now() + timedelta(seconds=5)
        frame_count = 0
        raw_frame = None
        success = False
        wipe_index = 100000
        on_wipe_end = None
        is_wiping = False
        def do_wipe_then(on_done):
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
                raw_frame = self.last_raw_frame
                success = True
            elif self.slowmo and frame_count % 3 != 0:
                raw_frame = self.last_raw_frame
                success = True
            else:
                success, raw_frame = self.readFrame()

            if success:
                self.last_raw_frame = raw_frame
                frame_time = self.cap.get_time()
                mini_view = self.capture_mini_view(raw_frame)
               
                # get the normal camera path
                camera, current_camera, next_camera = self.camera_path.get_camera_at_time(frame_time)
                if not is_wiping and writeOutputFile and current_camera == next_camera:
                    # no more cameras, finish rendering
                    def finish_up():
                        self.finished_rendering = True
                    do_wipe_then(finish_up) 

                # if we are saving a highlight, use the camera path from that highlight
                saving_highlight = self.highlights.get_active_save_highlight()
                if saving_highlight is not None:
                    camera, current_camera, next_camera = saving_highlight.get_camera_path().get_camera_at_time(frame_time)
               
                # get the cameras from a playing highlight
                if self.active_playback_highlight:
                    camera, current_camera, next_camera = self.active_playback_highlight.get_camera_path().get_camera_at_time(frame_time)

                use_live_cam = self.auto_record or self.isPaused() or frame_time > next_camera.time
                use_live_cam = use_live_cam and not self.active_playback_highlight
                zoom = self.zoom if use_live_cam else camera.zoom
                max_x = int(rotate_point(raw_frame.shape[0:2:1], self.rotation+self.angle_right)[1] - roi_size[0] / zoom)

                top_of_roi_adjust = np.interp([mouse_x], [0,max_x//2+self.center_x_offset, max_x], [self.top_of_roi_left, 0, self.top_of_roi_right])[0]

                x = int(np.clip(camera.x, 0, max_x)) # TODO - does this need to be clamped here?
                frame_y = int(camera.y)
                clamped_mouse_x = int(np.clip(mouse_x, 0, max_x))
                clamped_mouse_y = int(np.clip(mouse_y - 500, -500, 1500))

                frame_x = clamped_mouse_x if use_live_cam else x
                if zoom == 1.0:
                   clamped_mouse_y = 0

                angle = np.interp([frame_x], [0,max_x//2+self.center_x_offset, max_x], [self.angle_left, 0, self.angle_right])[0]
                frame = rotate_image(raw_frame, self.rotation+angle)
                live_y = (clamped_mouse_y - self.top_of_roi + top_of_roi_adjust)
                frame_y = live_y if use_live_cam else frame_y
                frame_y = int(rotate_point((frame_x + roi_size[0]//2 - frame.shape[1]//2, frame_y), -angle)[1])


                # ------------- cuts / wipes
                if frame_count == 1:
                    # Jump ahead to the first camera if it's far enough away
                    if current_camera.time - frame_time > 5000:
                        self.set_time(current_camera.time)
                        continue
                    # if we happen to be cutting to a camera as we also change files....
                    if current_camera.cut_to:
                        self.set_time(next_camera.time)
                        continue

                # handle iso selection by advancing to the next camera if the gap is > 30s
                if self.active_playback_highlight is None and iso and next_camera.time - frame_time > 30000:
                    self.set_time(next_camera.time)
                    continue

               # highlight start
                if not is_wiping:
                    if self.play_highlights and not saving_highlight and not self.pending_active_playback_highlight:
                        h = self.highlights.get_highlight_by_end_time(frame_time)
                        if h is not None:
                            self.pending_active_playback_highlight = h
                            def start_highlight():
                                h = self.pending_active_playback_highlight
                                self.active_playback_highlight = h
                                self.slowmo = h.slowmo
                                self.set_time(h.start_time)
                            do_wipe_then(start_highlight)
                                
                # highlight end
                if not is_wiping:
                    if self.active_playback_highlight and frame_time > self.active_playback_highlight.end_time:
                        def end_highlight():
                            _, at_wipe_end_cam, after_wipe_cam = self.camera_path.get_camera_at_time(frame_time)
                            if at_wipe_end_cam and at_wipe_end_cam.cut_to:
                                self.set_time(after_wipe_cam.time)
                            self.active_playback_highlight = None
                            self.pending_active_playback_highlight = None
                            self.slowmo = False
                        do_wipe_then(end_highlight)
                
                # camera cut
                if not is_wiping:
                    next_camera_good = (next_camera.time - frame_time > 2000) or current_camera == next_camera
                    if self.do_camera_cuts and not self.pending_active_playback_highlight and current_camera.cut_to and next_camera_good:
                        cut_to_time = next_camera.time
                        def jump_to_next_camera():
                            self.set_time(cut_to_time)
                        do_wipe_then(jump_to_next_camera)

                if mouse_click or (self.auto_record and frame_time > self.last_auto_time + 1000):
                    if self.auto_record:
                        self.last_auto_time = frame_time

                    if saving_highlight is not None:
                        saving_highlight.get_camera_path().add_camera_target(CameraTarget(frame_time, x=clamped_mouse_x, y=live_y, zoom=self.zoom))
                    else:
                        self.camera_path.add_camera_target(CameraTarget(frame_time, x=clamped_mouse_x, y=live_y, zoom=self.zoom))
                    mouse_click = False

                show_target = not writeOutputFile and zoom > 1.0 and (next_camera.time < frame_time or self.paused)
                outFrame = self._create_output_frame(frame, frame_x, frame_y, max_x, zoom, show_target)
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
                    (w,h) = (int(roi_size[0] / zoom), int(roi_size[1] / zoom)) # This is used for the rectangle drawing below

                    show_playback_rect = (not self.auto_record) and next_camera.time >= frame_time
                    if show_playback_rect:
                        blue = (255,0,0)
                        cv2.rectangle(frame, (x,frame_y),(x+w,frame_y+h), blue, 10)

                    if self.active_playback_highlight is not None:
                        highlight_cam,_,_ = self.active_playback_highlight.get_camera_path().get_camera_at_time(frame_time)
                        hx = int(np.clip(highlight_cam.x, 0, max_x))
                        cyan = (255,255,0)
                        cv2.rectangle(frame, (hx,frame_y),(hx+w,frame_y+h), cyan, 5)

                    if not show_playback_rect or self.isPaused():
                        purple = (255,0,200)
                        red = (55,0,200)
                        rect_col = purple if self.auto_record else red
                        cv2.rectangle(frame, (clamped_mouse_x,frame_y),(clamped_mouse_x+w,frame_y+h), rect_col, 10)
                        cv2.line(frame, (clamped_mouse_x+w//2,frame_y),(clamped_mouse_x+w//2,frame_y+h), rect_col, 1)
                        cv2.line(frame, (clamped_mouse_x+w//2-50,frame_y+h//2),(clamped_mouse_x+w//2+50,frame_y+h//2), rect_col, 1)

                    frame = frame[1200:2800,0:frame.shape[1]]
                    frame = cv2.resize(frame, view_size)
                    frame = rotate_image_crop(frame, -angle)

                    text_y = 100
                    line1 = f"{self.angle_left:.1f} | {self.rotation:.1f} | {self.angle_right:.1f} ({angle:.2f})| "
                    line1 += f"{self.camera_path.to_string(frame_time)} @ {ms_str(next_camera.time)} {len(self.highlights.get_highlights())} highlights"
                    if saving_highlight is not None:
                        line1 += f" Saving Highlight {saving_highlight.get_camera_path().to_string(frame_time)}"
                    if current_camera.cut_to:
                        line1 += " * IN CAMERA CUT *"
                    cv2.putText(frame, line1, (0,text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                    line2 = f"{ms_str(frame_time)} / {ms_str(total_vid_time)} Cap({self.cap.get_cap_index()}) {"Auto" if self.auto_record else ""} Zoom: {zoom:.1f}"
                    if self.play_highlights:
                        line2 += " will play highlights"
                    if self.highlights.get_highlight_at_time(frame_time):
                        line2 += " in highlight"
                    cv2.putText(frame, line2, (0,text_y+100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)


                    cv2.imshow('frame',frame)


                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                self.handleKeys(key, frame_time)
            else:
                # Break the loop if the end of the video is reached
                break
        finished = not self.isRunning()
        self.save_pickle(finished)
        # Release the video capture object and close the display window
        self.cap.release()
        if writeOutputFile:
            out.release()

        return finished

def load_logo():
    global logo_frames
    global logo_masks
    if args.logo is None:
        return
    cap = fastcap.FastCap([args.logo])
    while True:
        success, frame = cap.read_frame()
        if not success:
            break
        frame = cv2.resize(frame, full_out_size)
        logo_frames.append(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([153,15,85])
        upper = np.array([170,255,255])
        inverted_mask = cv2.inRange(hsv, lower, upper)
        logo_masks.append(inverted_mask)
    cap.release
 
def show_logo():
    raw_files = glob.glob(path.join(basePath, "*.mp4"))
    if len(raw_files) == 0:
        return
    cap2 = fastcap.FastCap(raw_files[0])
    
    for i,logo in enumerate(logo_frames):
        inverted_mask=logo_masks[i]
        mask = 255-inverted_mask

        _, frame = cap2.read_frame()
        frame = cv2.resize(frame,(1920,1080))
        bf = add_logo_to_frame(frame, i)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.imshow('final', bf)

    cap2.release()

def examine_files():
    raw_files = glob.glob(path.join(basePath, "*.mp4"))
    sources = [Path(f) for f in raw_files if not "_processed" in f]
    p = Processor([str(fp.with_suffix("")) for fp in sources])
    print(f"Starting process for {sources}")
    p.process()
    print(f"process for {sources} done")
 
load_logo()
if args.show_logo:
    show_logo()
else:
    examine_files()
cv2.destroyAllWindows()
