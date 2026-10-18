import glob
import math
import os
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
import gamestate
from highlight import Highlights
import logo_player
from util import rotate_image, rotate_image_crop, rotate_point
from review import (LEAD_IN_MS, PAUSE_AFTER_MS, REVIEW_JSON, ReviewSession, apply_goal, drop_known_goals, goal_bookmarks,
                    is_auto, legacy_goal_changes, refresh_path)

parser = argparse.ArgumentParser()
parser.add_argument("basepath", help="Base path to begin video processing from")
parser.add_argument("-r", "--render", action="store_true", help="Render the video")
parser.add_argument("-ns", "--no-skip", action="store_true", help="Don't skip files that are finished")
parser.add_argument("-i", "--iso", action="store_true", help="When ISO is active, skip forward to the next camera shot if the gap is > 30s")
parser.add_argument("--logo", type=str, help="Logo screen wipe")
parser.add_argument("--show-logo", action="store_true", help="Only show logo")
parser.add_argument("--pos", action="store_true", help="Show posession")
parser.add_argument("--keys", action="store_true", help="Show keys")
parser.add_argument("--review", action="store_true", help="Review the autotrack path, worst segments first")
parser.add_argument("--edit", action="store_true", help="Open for editing/recording without the start-up menu")
args = parser.parse_args()

def show_keys():
    print("Period Control")
    print("\tp/P - increment/decrement current period")

    print("Score Control")
    print("\t[{ - inc/dec left score")
    print("\t]} - inc/dec right score")
    print("\te - inc shot count")
    print("\tE - inc goal count")

    print("Camera Control")
    print("\tz/Z - zoom in/out")
    print("\tc   - hard camera cut")

    print("Highlights")
    print("\th - start/stop highlight")
    print("\tg - toggle highlight preview")
    print("\td - delete highlight")

    print("Frame Adjustment")
    print("\tx/X - dec/inc center offset")
    print("\to   - set zone offset")
    print("\tu/i - rotate left/right.  (can be done per rink end too)")
    print("\tj/k - move roi down/up")

    print("Mini-view Control")
    print("\tnm,. - move mini-view up/down/left/right")
    print("\tU/I - rotate mini-view left/right")

    print("Time Control")
    print("\tspace - pause")
    print("\t@ - jump to previous highlight")
    print("\t# - jump to next highlight")
    print("\ts  - slomo toggle")
    print("\tf  - fastmo toggle")
    print("\ta  - auto record toggle")
    print("\t`12 - move back 60s, 15s, 5s")
    print("\t345 - move forward 9s, 60s, 5m")
    print("\t8,9 - from current camera, move back/forward 1 canera mark")
    print("\t0 - move to last camera mark")

    print("Review (--review, or 1 in the start-up menu)")
    print("\t= / -  - next / previous segment (plays from 3s before, pauses after)")
    print("\tEnter  - segment OK / yes it's a goal / cut this intermission")
    print("\tBackspace - not a goal / keep this intermission")
    print("\tmiddle click, [ { ] } - add / take back a goal from here on")
    print("\tR      - switch between review and edit")
    print("\ta      - re-record: replaces the path only while recording")
    print("\tT      - start: rendering begins here (do this at the first faceoff)")
    print("\tY      - end: nothing after here is rendered")
    print("\tc ... c - cut from the first press to the second (C cancels)")

    print("Mouse Controls")
    print("\tleft   - place camera marker")
    print("\tright  - increase shot count (left/right auto detect from x value)")
    print("\tmiddle - increase goal count (left/right auto detect from x value)")

writeOutputFile = args.render
review_mode = args.review
basePath = args.basepath
show_posession = args.pos
preview = True
iso = args.iso
logo_manager = None

def scale(sz, s=0.25):
    return (int(sz[0]*s),int(sz[1]*s))

mouse_x = 0
mouse_y = 0
mouse_left_click = False
mouse_right_click = False
mouse_middle_click = False

view_size = (3340, 1050)
out_size = (1920,720)
full_out_size = (1920, 1080)
mini_view_size = (400, 400)
mini_view_rot_size = (350, 180)

roi_size = out_size

def ms_str(ms):
    h,m,s = str(timedelta(milliseconds=ms)).split(".")[0].split(":")
    if m.startswith("0"):
        m = m[1:]
    if h != "0":
        return f"{h}h{m}m{s}s"
    return f"{m}m{s}s"

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_left_click, mouse_middle_click, mouse_right_click
    mouse_x = int((x-out_size[0]/2)*2)
    mouse_y = int((y-out_size[1]/2)*2)
    match event:
        case cv2.EVENT_LBUTTONDOWN:
            mouse_left_click = True
        case cv2.EVENT_MBUTTONDOWN:
            mouse_middle_click = True
        case cv2.EVENT_RBUTTONDOWN:
            mouse_right_click = True

AUTO_PATH_PKL = "auto_path.pkl"  # written by autotrack.py

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

def convert_render(processed):
    """convert.bat's shrink (HEVC on the GPU), written next to the other finished games as
    <folder date>-<what you type>.mp4, e.g. 2026-10-18-Bombshells_v_Supernovas.mp4."""
    import subprocess
    game_dir = path.dirname(path.abspath(processed))
    prefix = path.basename(game_dir) + "-"
    name = input(f"name: {prefix}").strip()
    if not name:
        print("  cancelled")
        return
    if not name.lower().endswith((".mp4", ".mov")):
        name += ".mp4"
    out = path.join(path.dirname(game_dir), prefix + name)
    if path.exists(out) and input(f"  {out} exists; overwrite? [y/N] ").strip().lower() != "y":
        print("  cancelled")
        return
    print(f"  converting {processed} -> {out}")
    result = subprocess.run(["ffmpeg", "-y", "-i", processed, "-c:v", "hevc_nvenc", "-preset", "fast", "-vtag", "hvc1", out])
    size = lambda p: humanize.naturalsize(path.getsize(p))
    if result.returncode == 0:
        print(f"  done: {out} ({size(out)}, from {size(processed)})")
    else:
        print(f"  ffmpeg failed (exit {result.returncode})")

def start_menu():
    """With just a folder on the command line: say what's there and ask how to open it."""
    global writeOutputFile, review_mode
    import json
    import subprocess
    import sys
    while True:
        raw = sorted(f for f in glob.glob(path.join(basePath, "*.mp4")) if "_processed" not in f)
        if not raw:
            print(f"no .mp4 files in {basePath}")
            exit(1)
        game = load_pkl(str(Path(raw[0]).with_suffix("")) + ".pkl")
        auto = load_pkl(path.join(basePath, AUTO_PATH_PKL))
        review_json = path.join(basePath, REVIEW_JSON)
        entries = json.load(open(review_json))["entries"] if path.exists(review_json) else []
        done = set(game.get("review_done", [])) if game is not None else set()
        print(f"\n{basePath}: {len(raw)} videos")
        if game is not None:
            cp = game["camera_path"]
            ct = cp.camera_targets
            n_auto = sum(is_auto(c) for c in ct)
            line = f"  game pkl: {len(ct)} camera points ({n_auto} auto, {len(ct) - n_auto} recorded)"
            if ct and n_auto and any(not is_auto(c) and getattr(c, "source", "") == "human" for c in ct):
                line += f", starts {ms_str(cp.get_start_time())}, ends {ms_str(cp.get_last_cam_time())}"
            print(line)
        else:
            print("  game pkl: none yet (calibration defaults)")
        if auto is None:
            print("  autotrack: not run")
        else:
            print(f"  autotrack: auto path from {auto.get('autotrack', {}).get('created', '?')}"
                  + (f", {len(entries)} review items ({len(done & {e['t0'] for e in entries})} reviewed)" if entries else ""))
            if game is not None and ct and n_auto and auto.get("autotrack", {}).get("created") != game.get("auto_path_created"):
                print("    (newer than this game's copy: it is refreshed on opening, keeping your edits)")
        if auto is not None:
            print("  1) REVIEW the auto path, least certain parts first   [Enter]")
        else:
            print("  1) review (run autotrack first)")
        print("  2) edit / record (no review items shown; R switches to review)")
        print("  3) render")
        print("  4) run autotrack (about 50 min per game; re-runs take seconds)")
        processed = str(Path(raw[0]).with_suffix("")) + "_processed.mp4"
        if path.exists(processed):
            print(f"  5) convert the render for sharing -> {path.dirname(path.abspath(basePath))}\\<date>-<teams>.mp4")
        else:
            print("  5) convert the render for sharing (render first)")
        print("  q) quit")
        choice = input("> ").strip().lower()
        if choice == "" and auto is not None:
            choice = "1"
        if choice == "1" and auto is not None:
            review_mode = True
            return
        if choice == "2":
            return
        if choice == "3":
            writeOutputFile = True
            return
        if choice == "4":
            subprocess.run([sys.executable, path.join(path.dirname(path.abspath(__file__)), "autotrack.py"), basePath])
            continue
        if choice == "5" and path.exists(processed):
            convert_render(processed)
            continue
        if choice == "q":
            exit(0)

if not (args.render or args.review or args.edit or args.keys or args.show_logo):
    start_menu()

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback)

class Processor:
    def __init__(self, filenames):
        print(filenames)
        self.filenames = filenames
        self.filename = filenames[0]
        self.paused = not writeOutputFile
        self.auto_record = False
        self.last_auto_time = 0
        # Calibration defaults: the average of hand-calibrated games at the home rink
        # (kept in step with af/calibration.py DEFAULTS)
        self.mini_view_x = 3384
        self.mini_view_y = 2020
        self.zone_left = 793
        self.zone_right = 2129
        self.rotation = 37
        self.mini_rotation = 6.3
        self.top_of_roi = -1732
        self.top_of_roi_left = -75
        self.top_of_roi_right = 161
        self.had_pickle = False
        self.angle_left = -9
        self.angle_right = 9
        self.center_x_offset = 54
        self.last_raw_frame = None
        self.last_frame_time = 0
        self.slowmo = False
        self.fastmo = False
        self.zoom = 1.0
        self.bookmark_times_ms = []
        self.play_highlights = writeOutputFile
        self.do_camera_cuts = writeOutputFile
        self.finished_rendering = False
        self.pending_active_playback_highlight = None
        self.active_playback_highlight = None
        self.game_state = gamestate.GameState()
        loaded = load_pkl(self.filename + '.pkl')
        self.camera_path = CameraPath()
        self.highlights = Highlights()
        if loaded is not None:
            self.had_pickle = True
            self.load_camera_config_pickle(loaded)

            self.camera_path = loaded["camera_path"]
            self.highlights = loaded["highlights"]
            self.highlights.active_save_highlight = None

        self.review_done = loaded.get("review_done", []) if loaded is not None else []
        self.goal_changes = list(loaded.get("review_goal_changes", [])) if loaded is not None else []
        self.auto_path_created = loaded.get("auto_path_created") if loaded is not None else None

        # autotrack.py writes its path to a sidecar.  Take it while this game has no camera
        # path yet; when autotrack has been re-run since, refresh an auto path in place,
        # keeping this game's own edits (start/end, cuts, re-recordings, goal decisions).
        auto = load_pkl(path.join(path.dirname(self.filename), AUTO_PATH_PKL))
        if auto is not None:
            created = auto.get("autotrack", {}).get("created")
            targets = self.camera_path.camera_targets
            if len(targets) == 0:
                if loaded is None:
                    self.load_camera_config_pickle(auto)  # the calibration the path was computed with
                self.camera_path = auto["camera_path"]
                if not self.bookmark_times_ms:
                    self.bookmark_times_ms = list(auto.get("bookmark_times_ms", []))  # periods and goals
                self.auto_path_created = created
                print(f"Loaded {len(self.camera_path.camera_targets)} auto camera targets from {AUTO_PATH_PKL}")
            elif any(is_auto(c) for c in targets) and created != self.auto_path_created:
                import shutil
                backup = self.filename + ".pkl.before-refresh"
                shutil.copy2(self.filename + ".pkl", backup)
                print(f"Auto path re-run since this game was saved: refreshing (backup: {backup})")
                if loaded is not None and "review_goal_changes" not in loaded:
                    # goals typed in before they were recorded as changes: drop the ones the
                    # new path already has, or they'd count twice
                    self.goal_changes, dropped = drop_known_goals(legacy_goal_changes(targets),
                                                                  auto["camera_path"].camera_targets)
                    for t, team, _ in dropped:
                        print(f"  your goal at {ms_str(t)} is already in the auto path (scoreboard); not added twice")
                n_edits = sum(not is_auto(c) for c in targets)
                self.camera_path.camera_targets = refresh_path(targets, auto["camera_path"].camera_targets,
                                                               self.goal_changes)
                own = [b for b in self.bookmark_times_ms if not b[1].startswith(("goal", "period"))]
                periods = [b for b in auto.get("bookmark_times_ms", []) if b[1].startswith("period")]
                self.bookmark_times_ms = goal_bookmarks(self.camera_path.camera_targets, own + periods)
                self.auto_path_created = created
                final = self.camera_path.camera_targets[-1].game_state
                print(f"Refreshed the auto path from {AUTO_PATH_PKL} ({created}): kept {n_edits} edits and "
                      f"{len(self.goal_changes)} goal changes; score now {final.score[0]}-{final.score[1]}")

        self.review = None
        self.review_available = path.exists(path.join(path.dirname(self.filename), REVIEW_JSON))
        if review_mode:
            self.start_review()

    def start_review(self):
        self.review = ReviewSession(path.dirname(self.filename), self.review_done)
        # an intermission already cut (by hand, or before autotrack was re-run) needs no review
        cuts = [c.time for c in self.camera_path.camera_targets if c.cut_to]
        for e in self.review.entries:
            if e.get("kind") == "cut" and any(abs(t - e["t0"]) < 30000 for t in cuts):
                self.review.done.add(e["t0"])
        print(f"Review: {len(self.review.entries)} items in {REVIEW_JSON}")

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
        self.zone_left = pkl["zone_left"]
        self.zone_right = pkl["zone_right"]
        self.bookmark_times_ms = pkl["bookmark_times_ms"] if "bookmark_times_ms" in pkl else []


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
            "zone_left":self.zone_left,
            "zone_right":self.zone_right,
            ### end Camera config
            
            ### end Camera config
            # per file state
            "camera_path" : self.camera_path,
            "auto_record":self.auto_record,
            "highlights":self.highlights,
            "review_done": sorted(self.review.done) if self.review is not None else self.review_done,
            "review_goal_changes": self.goal_changes,
            "auto_path_created": self.auto_path_created,
            # processing state
            "rendered" : writeOutputFile and finished,
            "finished" : finished,
            "bookmark_times_ms" : self.bookmark_times_ms
        }
        save_pkl(self.filename+'.pkl', toSave)

    def set_time(self, time):
        self.auto_record = False
        self.cap.set_time(time)
        self.last_auto_time = 0
        self.last_raw_frame = None

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

    # No special state needs to be kept for zone times, we calculate it 
    # from camera position
    def calculate_zone_times_seconds(self):
        left,neutral,right = 0,0,0
        for c in self.camera_path.camera_targets:
            if c.time > self.last_frame_time:
                break
            # reverse sides in period 2
            if c.game_state.period == 2:
                if c.x < self.zone_left:
                    left += 1
                elif c.x > self.zone_right:
                    right += 1
                else:
                    neutral += 1
            else:
                if c.x < self.zone_left:
                    right += 1
                elif c.x > self.zone_right:
                    left += 1
                else:
                    neutral += 1

        return (left,neutral,right)

    def review_start_end(self):
        """(start ms, whether a recorded start exists, end ms) of the path being reviewed."""
        path_ = self.camera_path
        start_set = any(getattr(c, "source", "human") == "human" for c in path_.camera_targets)
        return path_.get_start_time(), start_set, path_.get_last_cam_time()

    def adjust_goal(self, t, team, delta):
        """Add (+1) or take back (-1) a goal for score index `team` from time t on, rewrite
        the goal bookmarks to match, and remember it so a refreshed auto path keeps it."""
        apply_goal(self.camera_path.camera_targets, t, team, delta)
        self.bookmark_times_ms = goal_bookmarks(self.camera_path.camera_targets, self.bookmark_times_ms)
        self.goal_changes.append((t, team, delta))

    def handle_review_keys(self, key, frame_time):
        """Review-mode keys; returns True when the key was handled here."""
        r = self.review
        cam, _, _ = self.camera_path.get_camera_at_time(frame_time)

        def marker(t, source, cut_to=False):
            c, _, nxt = self.camera_path.get_camera_at_time(t)  # the app takes game state from the next camera
            return CameraTarget(t, x=int(c.x), y=c.y, cut_to=cut_to, zoom=1.0, game_state=nxt.game_state, source=source)

        def cut(a, b):
            self.camera_path.replace_range(a - 500, b + 500, [marker(a, "review", cut_to=True), marker(b, "review")])

        if key in (ord('='), ord('-')):
            start, _, end = self.review_start_end()
            e = r.step(start, end, forward=key == ord('='))
            if e is not None and e.get("kind") == "cut":
                self.set_time(e["t0"])  # nothing to watch in an intermission: show where it starts
                self.paused = True
                r.at_end = True
            elif e is not None:
                self.set_time(max(0, e["t0"] - LEAD_IN_MS))
                self.paused = False
            return True
        e = r.current
        kind = e.get("kind", "camera") if e is not None else None
        if key == 13:  # Enter: fine / yes
            if kind == "cut" and e["t0"] not in r.done:
                cut(e["t0"], e["t1"])
            if kind == "goal" and not e.get("scored") and e["t0"] not in r.done:
                self.adjust_goal(e["t"], e["team"], +1)
                e["scored"] = True
            r.accept()
            return True
        if key == 8:  # Backspace: no
            if kind == "goal" and e.get("scored"):
                self.adjust_goal(e["t"], e["team"], -1)
                e["scored"] = False
            if kind in ("goal", "cut"):
                r.accept()
            return True
        if key == ord('T'):  # start: a recorded point here; earlier recorded points no longer count
            for c in self.camera_path.camera_targets:
                if c.time < frame_time and getattr(c, "source", "human") == "human":
                    c.source = "review"
            self.camera_path.replace_range(frame_time - 500, frame_time + 500, [marker(frame_time, "human")])
            return True
        if key == ord('Y'):  # end: nothing after here is rendered
            self.camera_path.replace_range(frame_time - 500, float("inf"), [marker(frame_time, "review")])
            return True
        if key == ord('c'):  # two presses: cut from the first to the second
            if r.cut_start is None:
                r.cut_start = frame_time
            elif frame_time > r.cut_start + 2000:
                cut(r.cut_start, frame_time)
                r.cut_start = None
            return True
        if key == ord('C'):
            r.cut_start = None
            return True
        if key == ord('a'):  # recording replaces the path from here until stopped
            self.auto_record = not self.auto_record
            if self.auto_record:
                self.paused = False
                r.record_from = frame_time - 1
            return True
        score_keys = {ord('['): ((1, 0), 1), ord('{'): ((1, 0), -1), ord(']'): ((0, 1), 1), ord('}'): ((0, 1), -1)}
        if key in score_keys:  # the score changes from here on, not just on the next recorded point
            side, delta = score_keys[key]
            self.adjust_goal(frame_time, self.team_for(side), delta)
            return True
        return False

    def team_for(self, side):
        """Score index for a (left, right) delta, as GameState.adjust_score maps it this period."""
        l, _ = self.game_state._get_lr(side)
        return 0 if l else 1

    def handle_keys(self, key, frame_time):
        if key == ord('R'):  # switch between review and edit without restarting
            if self.review is None:
                self.start_review()
            else:
                self.review_done = sorted(self.review.done)
                self.review = None
                self.auto_record = False
            return
        if self.review is not None and key != 255 and self.handle_review_keys(key, frame_time):
            return

        def set_camera_cut():
            highlight_is_saving = self.highlights.get_active_save_highlight() is not None
            if self.review is not None:
                return  # cuts are made with the two-press c in review mode
            if not highlight_is_saving and len(self.camera_path.camera_targets) > 0:
                self.camera_path.camera_targets[-1].cut_to = True
                self.auto_record = False

        # Period advance
        if key == ord('p'):
            # always set camera cut between periods
            set_camera_cut()

            self.game_state.period = self.game_state.period + 1
            self.bookmark_times_ms.append((frame_time, f"period - {self.game_state.period}"))
        if key == ord('P'):
            self.game_state.period = self.game_state.period - 1

        # score left
        if key == ord('['):
            self.game_state.adjust_score((1, 0))
        if key == ord('{'):
            self.game_state.adjust_score((-1, 0))
        # score right
        if key == ord(']'):
            self.game_state.adjust_score((0, 1))
        if key == ord('}'):
            self.game_state.adjust_score((0, -1))

        # shot
        if key == ord('E'):
            global mouse_middle_click
            mouse_middle_click = True
        if key == ord('e'):
            global mouse_right_click
            mouse_right_click = True

        #zoom 
        if key == ord('z'):
            self.zoom = min(3.0, self.zoom + 0.1)
        if key == ord('Z'):
            self.zoom = max(1.0, self.zoom - 0.1)

        # Hard camera cut
        if key == ord('c'):
            set_camera_cut()

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
                    self.camera_path.add_camera_target(CameraTarget(frame_time+1000, ct.x, ct.y, cut_to=True, zoom=1.0, game_state=self.game_state))
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

        # bookmarks
        if key == ord('b'): 
            self.bookmark_times_ms.append((frame_time, "bookmark"))

        if key == ord('g'):
            self.play_highlights = not self.play_highlights
            self.do_camera_cuts = not self.do_camera_cuts

        if key == ord('d'):
            self.highlights.delete_at_time(frame_time)

        if key == ord('s'):
            self.slowmo = not self.slowmo
            self.fastmo = False
        if key == ord('f'):
            self.fastmo = not self.fastmo
            self.slowmo = False

		# Center offset adjust
        if key == ord('x'):
            self.center_x_offset -= 1
        if key == ord('X'):
            self.center_x_offset += 1

        change_left = mouse_x < 500
        change_right = mouse_x > 3000

        if key == ord('o'):
            if mouse_x < 1500:
                self.zone_left = mouse_x
            else:
                self.zone_right = mouse_x

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
                self.top_of_roi += 5
        if key == ord('k'):
            if change_left:
                self.top_of_roi_left += 1
            elif change_right:
                self.top_of_roi_right += 1
            else:
                self.top_of_roi -= 5
        
        # Mini-view position
        if key == ord(','): # <
            self.mini_view_x -= 1
        if key == ord('.'): # >
            self.mini_view_x += 1
        if key == ord('n'):
            self.mini_view_y -= 1
        if key == ord('m'):
            self.mini_view_y += 1
        if key == ord('U'):
            self.mini_rotation -= 0.2
        if key == ord('I'):
            self.mini_rotation += 0.2

        # Time and recording controls
        if key == ord('a'):
            self.auto_record = not self.auto_record
            if self.auto_record:
                self.paused = False
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
        if key == ord('5'):
            self.skip_time(1000 * 60 * 5)

        # Camera controls
        if self.camera_path.has_targets():
            if key == ord('8'):
                self.set_time(self.camera_path.get_prev_cam_time(frame_time))
            if key == ord('9'):
                self.set_time(self.camera_path.get_next_cam_time(frame_time))
            if key == ord('0'):
                self.set_time(self.camera_path.get_last_cam_time())
                self.paused = True

    def readFrame(self):
        return self.cap.read_frame()

    def capture_mini_view(self, raw_frame):
        # Capture a 200x200 rectangle from a fixed location in the original frame.
        mini_view = raw_frame[self.mini_view_y:self.mini_view_y+mini_view_size[1], self.mini_view_x:self.mini_view_x+mini_view_size[0]]
        mini_view = rotate_image(mini_view, self.rotation+self.mini_rotation)
        dx,dy=75,130
        return mini_view[dy:dy+mini_view_rot_size[1],dx:dx+mini_view_rot_size[0]]
 

    def _create_output_frame(self, source_frame, x_pos, y_pos, zoom, show_target):
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

    def annotate_ouput_frame(self, out_frame):
        period = f"Period   {self.game_state.period}"
        cv2.putText(out_frame, period, (400,50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (215,215,215), 3)
        score_text = f"Score  {self.game_state.score[0]}-{self.game_state.score[1]}"
        cv2.putText(out_frame, score_text, (400,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (225,225,225), 3)
    
        # add on the score info
        if writeOutputFile and not show_posession:
            return

        score_text = f"Shots  {self.game_state.shots[0]}-{self.game_state.shots[1]}"
        cv2.putText(out_frame, score_text, (400,150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (225,225,225), 3)

        # add posession info
        l,n,r = self.calculate_zone_times_seconds()
        names = f"  Left  | Neutral | Right: (Ozone time, bench side)"
        cv2.putText(out_frame, names, (1200,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (215,215,215), 2)
        times = f"{ms_str(l*1000)} | {ms_str(n*1000)} | {ms_str(r*1000)}"
        cv2.putText(out_frame, times, (1200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (215,215,215), 2)

    def truncate_bookmarks_to_time(self, frame_time_ms):
        if len(self.bookmark_times_ms) == 0:
            return
        self.bookmark_times_ms.sort(key=lambda x: x[0])
        index = 0
        while index < len(self.bookmark_times_ms):
            if self.bookmark_times_ms[index][0] > frame_time_ms:
                break
            index = index + 1
        self.bookmark_times_ms = self.bookmark_times_ms[0:index]

    def process(self):
        global mouse_x
        global mouse_left_click, mouse_middle_click, mouse_right_click
        global logo_manager
        self.cap = fastcap.FastCap([x + ".mp4" for x in self.filenames])

        out = None
        bookmark_out_times_ms = []
        next_bookmark_index_to_find = 0
        last_out_time_ms = 0

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

        while self.isRunning():
            frame_count = frame_count + 1

            if self.isPaused() and self.last_raw_frame is not None:
                raw_frame = self.last_raw_frame
                success = True
            elif self.slowmo and frame_count % 3 != 0:
                raw_frame = self.last_raw_frame
                success = True
            else:
                success, raw_frame = self.readFrame()
                # just read two frames for fastmo
                if self.fastmo:
                    success, raw_frame = self.readFrame()

            if success:
                if raw_frame is None:
                    print("This should never happen.  But it has.")
                    continue

                self.last_raw_frame = raw_frame
                frame_time = self.cap.get_time()
                self.last_frame_time = frame_time
                mini_view = self.capture_mini_view(raw_frame)
               
                # get the normal camera path
                camera, current_camera, next_camera = self.camera_path.get_camera_at_time(frame_time)
                no_future_cameras = current_camera == next_camera
                if not no_future_cameras:
                    self.game_state = next_camera.game_state.copy()

                if not logo_manager.is_playing() and writeOutputFile and no_future_cameras:
                    # no more cameras, finish rendering
                    def finish_up():
                        self.finished_rendering = True
                    logo_manager.do_wipe_then(finish_up) 

                # if we are saving a highlight, use the camera path from that highlight
                active_save_highlight = self.highlights.get_active_save_highlight()
                if active_save_highlight is not None:
                    camera, current_camera, next_camera = active_save_highlight.get_camera_path().get_camera_at_time(frame_time)
               
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
                    # Jump ahead to the start (the first camera, or the first recorded one on an
                    # autotrack path) if it's far enough away
                    start_time = self.camera_path.get_start_time()
                    if start_time - frame_time > 5000:
                        self.set_time(start_time)
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
                if not logo_manager.is_playing():
                    if self.play_highlights and not active_save_highlight and not self.pending_active_playback_highlight:
                        h = self.highlights.get_highlight_by_end_time(frame_time)
                        if h is not None:
                            self.pending_active_playback_highlight = h
                            def start_highlight():
                                h = self.pending_active_playback_highlight
                                self.active_playback_highlight = h
                                self.slowmo = h.slowmo
                                self.set_time(h.start_time)
                            logo_manager.do_wipe_then(start_highlight)
                                
                # highlight end
                if not logo_manager.is_playing():
                    if self.active_playback_highlight and frame_time > self.active_playback_highlight.end_time:
                        def end_highlight():
                            _, at_wipe_end_cam, after_wipe_cam = self.camera_path.get_camera_at_time(frame_time)
                            if at_wipe_end_cam and at_wipe_end_cam.cut_to:
                                self.set_time(after_wipe_cam.time)
                            self.active_playback_highlight = None
                            self.pending_active_playback_highlight = None
                            self.slowmo = False
                        logo_manager.do_wipe_then(end_highlight)
                
                # camera cut
                if not logo_manager.is_playing():
                    next_camera_good = (next_camera.time - frame_time > 2000) # or no_future_cameras # this OR makes no sense......
                    if self.do_camera_cuts and not self.pending_active_playback_highlight and current_camera.cut_to and next_camera_good:
                        cut_to_time = next_camera.time
                        def jump_to_next_camera():
                            self.set_time(cut_to_time)
                        logo_manager.do_wipe_then(jump_to_next_camera)

                mouse_click = mouse_left_click or mouse_right_click or mouse_middle_click
                if self.review is not None and (mouse_middle_click or mouse_right_click) and not mouse_left_click:
                    # review: a goal counts from here on (and survives a refreshed auto path);
                    # neither click moves the camera
                    if mouse_middle_click:
                        attribute_to_left_team = clamped_mouse_x > (max_x//2 + self.center_x_offset)
                        self.adjust_goal(frame_time, self.team_for((1, 0) if attribute_to_left_team else (0, 1)), +1)
                    mouse_middle_click = mouse_right_click = mouse_click = False
                if mouse_click or (self.auto_record and frame_time > self.last_auto_time + 1000):
                    attribute_to_left_team = clamped_mouse_x > (max_x//2 + self.center_x_offset)
                    if mouse_right_click:
                        self.game_state.adjust_shots((1,0) if attribute_to_left_team else (0,1))
                        self.bookmark_times_ms.append((frame_time, f"shot {self.game_state.shots[0]}-{self.game_state.shots[1]}"))
                    if mouse_middle_click:
                        self.game_state.adjust_score((1,0) if attribute_to_left_team else (0,1))
                        self.bookmark_times_ms.append((frame_time, f"goal {self.game_state.score[0]}-{self.game_state.score[1]}"))

                    if self.auto_record:
                        self.last_auto_time = frame_time

                    new_target = CameraTarget(frame_time, x=clamped_mouse_x, y=live_y, zoom=self.zoom, game_state=self.game_state)
                    if active_save_highlight is not None:
                        active_save_highlight.get_camera_path().add_camera_target(new_target)
                    elif self.review is not None:
                        # review: replace only what was re-recorded, keep the path after it
                        new_target.source = "review"
                        if self.auto_record and self.review.record_from is not None:
                            self.camera_path.replace_range(self.review.record_from, frame_time, [new_target])
                            self.review.record_from = frame_time + 1
                        else:
                            self.camera_path.replace_range(frame_time - 500, frame_time + 500, [new_target])
                    else:
                        self.camera_path.add_camera_target(new_target)
                        self.truncate_bookmarks_to_time(frame_time)

                    mouse_left_click = False
                    mouse_middle_click = False
                    mouse_right_click = False

                # review: stop just after the segment so it can be accepted or re-recorded
                r = self.review
                if r is not None and r.current is not None and not r.at_end and not self.auto_record \
                        and not self.paused and frame_time > r.current["t1"] + PAUSE_AFTER_MS:
                    self.paused = True
                    r.at_end = True

                show_target = not writeOutputFile and zoom > 1.0 and (next_camera.time < frame_time or self.paused)
                out_frame = self._create_output_frame(frame, frame_x, frame_y, zoom, show_target)
                # Composite the captured mini-view onto the output frame at (0,0)
                out_frame[0:mini_view_rot_size[1], 0:mini_view_rot_size[0]] = mini_view

                self.annotate_ouput_frame(out_frame)
                out_frame = logo_manager.add_logo_if_needed(out_frame)

                if writeOutputFile:
                    if datetime.now() > next_print:
                        delta = datetime.now() - start_time
                        print(f"Processed {frame_count} frames in {delta} ({frame_count / delta.seconds} fps)")
                        next_print = datetime.now() + timedelta(seconds=30)

                    if len(self.bookmark_times_ms) > next_bookmark_index_to_find:
                        mark = self.bookmark_times_ms[next_bookmark_index_to_find]
                        if mark[0] < frame_time:
                            bookmark_out_times_ms.append((last_out_time_ms, mark[1]))
                            next_bookmark_index_to_find += 1

                    out.write(out_frame)
                    last_out_time_ms += frame_ms

                if preview:
                    cv2.imshow('preview',out_frame)
                
                if not writeOutputFile:
                    # Display the annotated main frame
                    (w,h) = (int(roi_size[0] / zoom), int(roi_size[1] / zoom)) # This is used for the rectangle drawing below

                    def draw_target_rect(frame, x, y, color):
                        cv2.rectangle(frame, (x,y),(x+w,y+h), color, 10)
                        cv2.line(frame, (x+w//2,y),(x+w//2,y+h), color, 1)
                        cv2.line(frame, (x+w//2-50,y+h//2),(x+w//2+50,y+h//2), color, 1)

                    show_playback_rect = (not self.auto_record) and next_camera.time >= frame_time
                    if show_playback_rect:
                        blue = (255,0,0)
                        draw_target_rect(frame, x,frame_y, blue)

                    if self.active_playback_highlight is not None:
                        highlight_cam,_,_ = self.active_playback_highlight.get_camera_path().get_camera_at_time(frame_time)
                        cyan = (255,255,0)
                        draw_target_rect(frame, int(highlight_cam.x),frame_y, cyan)

                    purple,red = (255,0,200),(55,0,200)
                    rect_col = purple if self.auto_record else red
                    draw_target_rect(frame, clamped_mouse_x,frame_y, rect_col)

                    if len(self.camera_path.camera_targets) == 0:
                        cv2.line(frame, (self.zone_left+w//2,0),(self.zone_left+w//2,frame.shape[0]), (255,255,255), 1)
                        cv2.line(frame, (self.zone_right+w//2,0),(self.zone_right+w//2,frame.shape[0]), (255,255,255), 1)

                    # reshape and rotate the large window back so it has stable rotation
                    frame = frame[1200:2800,0:frame.shape[1]]
                    frame = cv2.resize(frame, view_size)
                    frame = rotate_image_crop(frame, -angle)

                    text_y = 30
                    line1 = f"{self.angle_left:.1f} | {self.rotation:.1f} | {self.angle_right:.1f} ({angle:.2f})| "
                    line1 += f"{self.camera_path.to_string(frame_time)} @ {ms_str(next_camera.time)} P:{self.game_state.period} {len(self.highlights.get_highlights())} highlights {len(self.bookmark_times_ms)} bookmarks"
                    if active_save_highlight is not None:
                        line1 += f" Saving Highlight {active_save_highlight.get_camera_path().to_string(frame_time)}"
                    if current_camera.cut_to:
                        line1 += " * IN CAMERA CUT *"
                    if self.fastmo:
                        line1 += " FASTMO"
                    if self.slowmo:
                        line1 += " SLOWMO"
                    cv2.putText(frame, line1, (0,text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                    line2 = f"{ms_str(frame_time)} / {ms_str(total_vid_time)} Cap({self.cap.get_cap_index()}) {"Auto" if self.auto_record else ""} Zoom: {zoom:.1f}"
                    if self.play_highlights:
                        line2 += " will play highlights"
                    if self.highlights.get_highlight_at_time(frame_time):
                        line2 += " in highlight"
                    cv2.putText(frame, line2, (0,text_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                    left,neutral,right = self.calculate_zone_times_seconds()
                    line3 = f"{ms_str(left*1000)} | {ms_str(neutral*1000)} | {ms_str(right*1000)}"
                    cv2.putText(frame, line3, (0,text_y+50*2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                    if self.review is not None:
                        start, start_set, end = self.review_start_end()
                        for k, line in enumerate(self.review.status_lines(frame_time, start, start_set, end)):
                            cv2.putText(frame, line, (0, text_y + 50 * (3 + k)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        # under the status lines: the window can be taller than the screen
                        self.review.draw_timeline(frame, frame_time, total_vid_time, start, end, text_y + 50 * 5 - 20)
                    elif self.review_available:
                        cv2.putText(frame, "review items available: R for review mode", (0, text_y + 50 * 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                    cv2.imshow('frame',frame)


                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                self.handle_keys(key, frame_time)
            else:
                # Break the loop if the end of the video is reached
                break
        finished = not self.isRunning()
        self.save_pickle(finished)
        # Release the video capture object and close the display window
        self.cap.release()
        if writeOutputFile:
            out.release()
            with open(path.join(basePath, self.filename+"_bookmarks.txt"), "w") as bookmark_file:
                for b in bookmark_out_times_ms:
                    bookmark_file.write(f"https://www.youtube.com/watch?v=fileguid&t={int(b[0]/1000)-5}s  -  {b[1]}\n")

        return finished

def load_logo():
    global logo_manager
    logo_manager = logo_player.LogoPlayer(args.logo, full_out_size=full_out_size)
 
def show_logo():
    raw_files = glob.glob(path.join(basePath, "*.mp4"))
    if len(raw_files) == 0:
        return
    cap2 = fastcap.FastCap(raw_files)

    running = True
    def on_done():
        nonlocal running
        running = False
    logo_manager.do_wipe_then(on_done)
    
    while running:
        running, frame = cap2.read_frame()
        frame = cv2.resize(frame,(1920,1080))
        bf = logo_manager.add_logo_if_needed(frame)

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
    print(f"sharks_render.bat {os.path.dirname(sources[0])}")
    print(f"python sportscam {os.path.dirname(sources[0])} --render")

show_keys()
if args.keys:
    exit(0)
 
load_logo()
if args.show_logo:
    show_logo()
else:
    examine_files()
cv2.destroyAllWindows()
