#!/usr/bin/env python
"""GPU ASR via mlx-whisper. Usage: mlx_transcribe.py <audio.wav> <out.json> [model]"""
import sys, json, time
import mlx_whisper

audio = sys.argv[1]
out = sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "mlx-community/whisper-large-v3-turbo"

t0 = time.time()
res = mlx_whisper.transcribe(
    audio, path_or_hf_repo=model, language="en",
    word_timestamps=True, verbose=False,
    condition_on_previous_text=False,   # avoids runaway repetition on long files
)
dt = time.time() - t0
segs = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in res["segments"]]
json.dump({"segments": segs, "text": res.get("text", ""), "asr_seconds": dt}, open(out, "w"), ensure_ascii=False, indent=1)
audio_dur = segs[-1]["end"] if segs else 0
print(f"ASR done in {dt:.1f}s for {audio_dur:.0f}s audio  ->  {audio_dur/max(dt,1):.1f}x realtime")
print(f"segments: {len(segs)}")
