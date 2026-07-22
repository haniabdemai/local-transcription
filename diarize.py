#!/usr/bin/env python
"""Speaker diarization via pyannote, bypassing torchcodec by pre-loading the waveform.
Usage: diarize.py <audio.wav> <out.json> [max_speakers]"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import torch, soundfile as sf
from pyannote.audio import Pipeline

audio = sys.argv[1]; out = sys.argv[2]
max_spk = int(sys.argv[3]) if len(sys.argv) > 3 else None

t0 = time.time()
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
try:
    pipe.to(torch.device(dev))
except Exception:
    dev = "cpu"; pipe.to(torch.device("cpu"))

# load wav ourselves (16k mono) -> (channel, time) tensor; avoids pyannote's torchcodec reader
wav, sr = sf.read(audio, dtype="float32")
if wav.ndim == 1:
    wav = wav[None, :]
else:
    wav = wav.T
waveform = torch.from_numpy(wav)
inp = {"waveform": waveform, "sample_rate": sr}

kwargs = {}
if max_spk: kwargs["max_speakers"] = max_spk
try:
    diar = pipe(inp, **kwargs)
except Exception as e:
    # MPS sometimes fails on an op; retry on CPU
    dev = "cpu"; pipe.to(torch.device("cpu"))
    diar = pipe(inp, **kwargs)

# pyannote 4.x community-1 returns DiarizeOutput; use exclusive (non-overlapping) turns
annotation = getattr(diar, "exclusive_speaker_diarization", None) or getattr(diar, "speaker_diarization", diar)
turns = [{"start": float(t.start), "end": float(t.end), "speaker": spk}
         for t, _, spk in annotation.itertracks(yield_label=True)]
dt = time.time() - t0
dur = turns[-1]["end"] if turns else 0
json.dump({"turns": turns, "device": dev, "diar_seconds": dt}, open(out, "w"), indent=1)
print(f"Diarization ({dev}) done in {dt:.1f}s for {dur:.0f}s audio -> {dur/max(dt,1):.1f}x realtime")
print(f"speakers: {sorted(set(t['speaker'] for t in turns))}  turns: {len(turns)}")
