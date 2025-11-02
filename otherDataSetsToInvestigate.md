from datasets import load_dataset
import soundfile as sf
import pandas as pd
from pathlib import Path

OUT = Path("meddialog_audio_export")
OUT_AUDIO = OUT / "wav"
OUT_AUDIO.mkdir(parents=True, exist_ok=True)

manifest_rows = []

ds = load_dataset("aline-gassenn/MedDialog-Audio_v2", split="train")
# you can repeat for "validation"/"test" if they exist

for i, row in enumerate(ds):
    # row should have something like row["audio"]["array"], row["audio"]["sampling_rate"], row["text"]
    audio_arr = row["audio"]["array"]
    sr = row["audio"]["sampling_rate"]

    wav_path = OUT_AUDIO / f"clip_{i:06d}.wav"
    sf.write(str(wav_path), audio_arr, sr)

    manifest_rows.append({
        "id": f"clip_{i:06d}",
        "wav_file": str(wav_path),
        "sample_rate": sr,
        "transcript": row.get("text", "")
    })

pd.DataFrame(manifest_rows).to_csv(OUT / "manifest.csv", index=False)



