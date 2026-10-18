# Autoframe - AI instructions
## Overview
This project takes a wide angled video stream that is deliberately rotated so that the ice hockey
rink fits diagonal corner to corner.  The purpose of the app is to convert the wide view
into a smaller viewport that is better for watching on youtube.  As it is, the camera movement
is *entirely* human controlled.

## Goal
I want to automate as much of this process as I possibly can.  Ideally, I point a script at a folder 
that has raw data and I get the same result back as a human.

## Existing work
The `autoframe` branch contains changes that I have worked with an AI on to try and improve the situation,
you should understand this.

## How to work
* The folder `D:\RawVideo\autoframe` contains a correctly framed and human-tracked video in the pkl file.  
Use this as baseline
* Build test loops that can be validated against the baseline in short runs.  When the autoframing is 
matching baseline well, extend the run length
* Work in one or more branches (*not* main)
* I would like you to work without much input for me, try multiple approaches (in difference branches as needed)
* Write frequent progress reports to the docs folder so I can monitor what approaches are being tried

## Models
* All work runs on **Claude Opus 5.5** (no Fable credits available).
* If an approach is essentially stuck and the next step seems to need deeper reasoning than Opus
  is providing, stop that line of evaluation. Mark it in its progress note and in `INDEX.md` as
  **"BLOCKED – Fable 5.1 recommended"**, with a short description of what Fable should look at.
  Then move on to other approaches.

## Decisions (from Q&A, 2026-10-18)
* The single pkl (`DJI_20260822110551_0015_D.pkl`) covers all four .mp4 files in the folder (one game).
* Scope: camera path only (X, and zoom if it varies). No goals, periods, shots, or possession.
* Target: autoframe X values should closely match the human X values. Claude picks the exact metric.
* Fully automatic is preferred. Manual per-game calibration is acceptable. The example data already has
  valid calibration for rotation and viewport heights.
* Scratch files (proxies, caches, detections, debug video) go in `D:\temp`.
* Branches stay local, never pushed. Claude decides what to do with untracked files.
* Autonomy: keep working without input. Pause at good stopping points or for crucial questions, and
  pursue other approaches while waiting. Installing packages is fine.

## Progress notes
* Notes live in `docs/progress/` in the main checkout (`C:\dev\sportscam`), so they can be read
  without switching branches. Experiment branches are worked on in separate git worktrees, so the
  main checkout never changes branch.
* `docs/progress/INDEX.md`: one line per approach (branch, status, best score), newest first.
* `docs/progress/<branch>.md`, one per approach: the plan, written *before* starting; dated progress
  entries; then a summary at each stopping point.
* When an approach is kept, its notes are also committed to its branch.