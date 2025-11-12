@REM ffmpeg -i .\DJI_20251109190622_0004_D.MP4 -c:v hevc_nvenc -preset fast -vtag hvc1 4.mp4
ffmpeg -i %1 -c:v hevc_nvenc -preset fast -vtag hvc1 %2