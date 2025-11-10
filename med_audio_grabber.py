#!/usr/bin/env python3
import os
import sys
import json
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager
from datetime import datetime
from tqdm import tqdm

import pandas as pd
import soundfile as sf

from datasets import load_dataset, Audio
from pydub import AudioSegment

# ----------------------------
# CONFIG: Add/remove datasets here
# ----------------------------
DATASETS: List[Dict] = [
    # Kaggle datasets
    {
        "kind": "kaggle",
        "id": "ammarshafiq/healthcare-appointment-booking-calls-dataset",  # "Our Clinic"
        "name": "our_clinic_calls",
        "transcript_globs": ["*.txt", "*.csv", "*.tsv", "*.json"],
    },
    {
        "kind": "kaggle",
        "id": "azmayensabil/doctor-patient-conversation-large",
        "name": "doctor_patient_conversation_large",
        "transcript_globs": ["*.txt", "*.csv", "*.tsv", "*.json"],
    },

    # Hugging Face datasets
    {
        "kind": "hf",
        "id": "aline-gassenn/MedDialog-Audio_v2",
        "name": "meddialog_audio_v2",
        "splits": ["train"],  # add "validation","test" if present
        "audio_col": "audio",
        "text_col": "text",
    },
    {
        "kind": "hf",
        "id": "united-we-care/United-Syn-Med",
        "name": "united_syn_med",
        "splits": ["train"],
        "audio_col": "audio",
        "text_col": "text",
    },
]

# Global audio normalization
TARGET_SR = 16000
TARGET_CH = 1

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def have_kaggle_creds() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists()

def run_cmd(cmd: List[str]):
    subprocess.run(cmd, check=True)

def write_csv_atomic(df: pd.DataFrame, out_path: Path):
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)

def duration_sec_from_wav(path: Path) -> Optional[float]:
    try:
        info = sf.info(str(path))
        return round(float(info.frames) / float(info.samplerate), 3)
    except Exception:
        return None

def convert_any_to_wav(src_path: Path, dst_path: Path, sr: int = TARGET_SR, ch: int = TARGET_CH) -> Optional[float]:
    # If destination exists, skip conversion and compute duration from dst
    if dst_path.exists():
        return duration_sec_from_wav(dst_path)

    audio_seg = AudioSegment.from_file(src_path)
    audio_seg = audio_seg.set_frame_rate(sr).set_channels(ch)
    ensure_dir(dst_path.parent)
    audio_seg.export(dst_path, format="wav")
    return round(len(audio_seg) / 1000.0, 3)

def find_first_matching(base: Path, patterns: List[str], stem_lower: str) -> Optional[Path]:
    for pat in patterns:
        for p in base.rglob(pat):
            if p.stem.lower() == stem_lower:
                return p
    return None

@contextmanager
def dataset_lock(ds_dir: Path):
    lock = ds_dir / ".inprogress"
    if lock.exists():
        print(f"[i] Resuming dataset with existing lock: {lock}")
    else:
        ensure_dir(ds_dir)
        lock.write_text(datetime.utcnow().isoformat() + "Z")
    try:
        yield
    finally:
        # Remove lock only if processing completed without unhandled exception
        if lock.exists():
            lock.unlink(missing_ok=True)

def rebuild_manifest_from_disk(
    dataset_name: str,
    audio_wav_dir: Path,
    audio_raw_dir: Optional[Path],
    transcripts_dir: Path,
    transcript_globs: Optional[List[str]] = None,
    include_inline_transcripts: bool = True,
) -> pd.DataFrame:
    rows = []
    for wav in sorted(audio_wav_dir.glob("*.wav")):
        stem = wav.stem
        tfile = None
        ttext = ""
        if transcript_globs:
            tfile_path = find_first_matching(transcripts_dir, transcript_globs, stem.lower())
            if tfile_path:
                tfile = str(tfile_path)
        if include_inline_transcripts and not tfile:
            # HF exports may have paired .txt we created; grab it if present
            candidate = transcripts_dir / f"{stem}.txt"
            if candidate.exists():
                tfile = str(candidate)
        if tfile and not ttext:
            try:
                ttext = Path(tfile).read_text(encoding="utf-8").strip()
            except Exception:
                ttext = ""

        orig_audio = ""
        if audio_raw_dir is not None:
            # Best-effort: original file may have different extension; try common ones
            for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
                cand = audio_raw_dir / f"{stem}{ext}"
                if cand.exists():
                    orig_audio = str(cand)
                    break
        else:
            orig_audio = str(wav)

        dur = duration_sec_from_wav(wav) or 0.0

        rows.append({
            "dataset": dataset_name,
            "id": stem,
            "wav_file": str(wav),
            "orig_audio": orig_audio,
            "sample_rate": TARGET_SR,
            "channels": TARGET_CH,
            "duration_sec": dur,
            "transcript": ttext,
            "transcript_file": tfile or "",
        })

    return pd.DataFrame(rows)

# ----------------------------
# Kaggle path (resumable)
# ----------------------------
def process_kaggle(entry: Dict, out_root: Path, force: bool = False, rebuild_manifest: bool = False) -> Path:
    if not have_kaggle_creds():
        print("ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json", file=sys.stderr)
        sys.exit(1)

    name = entry["name"]
    slug = entry["id"]

    ds_dir = out_root / name
    raw_dir = ds_dir / "raw"
    audio_raw = ds_dir / "audio_raw"
    audio_wav = ds_dir / "audio_wav"
    transcripts = ds_dir / "transcripts"
    manifest = ds_dir / "manifest.csv"

    ensure_dir(ds_dir); ensure_dir(raw_dir); ensure_dir(audio_raw); ensure_dir(audio_wav); ensure_dir(transcripts)

    if manifest.exists() and not (force or rebuild_manifest):
        print(f"[→] {name}: manifest exists; skipping (use --rebuild-manifest or --force to regenerate).")
        return ds_dir

    with dataset_lock(ds_dir):
        # Download ZIP (skip if already present)
        zips_before = set(ds_dir.glob("*.zip"))
        cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(ds_dir)]
        if force:
            cmd.append("--force")
        print(f"[+] Kaggle download: {slug}")
        run_cmd(cmd)
        zips_after = set(ds_dir.glob("*.zip"))
        for z in sorted(zips_after):
            print(f"    found zip: {z.name}")

        # Extract any ZIPs into raw/ (skip files that already exist)
        for z in sorted(zips_after - zips_before) | (zips_after & zips_before):
            print(f"    extracting: {z.name}")
            with zipfile.ZipFile(z, "r") as zf:
                for m in zf.infolist():
                    target = raw_dir / m.filename
                    if target.exists() and not force:
                        continue
                    zf.extract(m, raw_dir)

        # Organize files (idempotent)
        audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
        text_exts  = {".txt", ".csv", ".tsv", ".json", ".xlsx"}

        for p in raw_dir.rglob("*"):
            if p.is_dir():
                continue
            # Already in right place?
            if p.is_relative_to(audio_raw) or p.is_relative_to(transcripts) or p.is_relative_to(audio_wav):
                continue

            ext = p.suffix.lower()
            try:
                if ext in audio_exts:
                    dest = audio_raw / p.name
                    if not dest.exists() or force:
                        ensure_dir(dest.parent)
                        p.replace(dest) if not dest.exists() else None
                elif ext in text_exts:
                    dest = transcripts / p.name
                    if not dest.exists() or force:
                        ensure_dir(dest.parent)
                        p.replace(dest) if not dest.exists() else None
            except Exception as e:
                print(f"[!] Move failed for {p}: {e}")

        # Convert audio → WAV (skip existing)
        audio_files = sorted([p for p in audio_raw.glob("*") if p.is_file()])
        for src in tqdm(audio_files, desc=f"{name}: converting to WAV"):
            dst = audio_wav / f"{src.stem}.wav"
            try:
                _ = convert_any_to_wav(src, dst, TARGET_SR, TARGET_CH)  # duration computed if new
            except Exception as e:
                print(f"[!] Conversion failed for {src}: {e}")

        # Build manifest by scanning disk (robust to partials)
        df = rebuild_manifest_from_disk(
            dataset_name=name,
            audio_wav_dir=audio_wav,
            audio_raw_dir=audio_raw,
            transcripts_dir=transcripts,
            transcript_globs=entry.get("transcript_globs", None),
        )
        write_csv_atomic(df, manifest)
        print(f"[✓] Wrote {manifest}")

    return ds_dir

# ----------------------------
# HF path (resumable)
# ----------------------------
def process_hf(entry: Dict, out_root: Path, force: bool = False, rebuild_manifest: bool = False) -> Path:
    name = entry["name"]
    ds_id = entry["id"]
    splits = entry.get("splits", ["train"])
    audio_col = entry.get("audio_col", "audio")
    text_col  = entry.get("text_col", "text")

    ds_dir = out_root / name
    audio_wav = ds_dir / "audio_wav"
    transcripts = ds_dir / "transcripts"
    manifest = ds_dir / "manifest.csv"

    ensure_dir(ds_dir); ensure_dir(audio_wav); ensure_dir(transcripts)

    # If fully processed and not forcing, skip heavy work
    if manifest.exists() and not (force or rebuild_manifest):
        print(f"[→] {name}: manifest exists; skipping export (use --rebuild-manifest or --force).")
        return ds_dir

    with dataset_lock(ds_dir):
        for split in splits:
            print(f"[+] HF load: {ds_id} [{split}]")
            ds = load_dataset(ds_id, split=split)  # cached by HF under ~/.cache
            # Cast to our target SR; HF will resample on the fly
            ds = ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SR))

            # Export (skip existing wavs)
            for i, row in enumerate(tqdm(ds, desc=f"{name}:{split} export")):
                clip_id = f"{split}_{i:06d}"
                wav_path = audio_wav / f"{clip_id}.wav"
                if wav_path.exists() and not force:
                    # ensure transcript exists too (create if missing)
                    transcript_val = row.get(text_col, "")
                    tfile = transcripts / f"{clip_id}.txt"
                    if transcript_val and (not tfile.exists() or force):
                        try:
                            tfile.write_text(str(transcript_val), encoding="utf-8")
                        except Exception as e:
                            print(f"[!] Transcript write failed ({tfile}): {e}")
                    continue

                try:
                    audio = row[audio_col]  # {"array": np.array, "sampling_rate": TARGET_SR}
                    sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
                    transcript_val = row.get(text_col, "")
                    if transcript_val:
                        tfile = transcripts / f"{clip_id}.txt"
                        tfile.write_text(str(transcript_val), encoding="utf-8")
                except Exception as e:
                    print(f"[!] Export failed for {clip_id}: {e}")

        # Manifest from disk
        df = rebuild_manifest_from_disk(
            dataset_name=name,
            audio_wav_dir=audio_wav,
            audio_raw_dir=None,
            transcripts_dir=transcripts,
            transcript_globs=None,
            include_inline_transcripts=True,
        )
        write_csv_atomic(df, manifest)
        print(f"[✓] Wrote {manifest}")

    return ds_dir

# ----------------------------
# Master runner
# ----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Resumable downloader/indexer for medical conversation audio (Kaggle + Hugging Face).")
    ap.add_argument("--out", default="med_audio_data", help="Output root directory")
    ap.add_argument("--only", nargs="*", help="Limit to dataset names in config (e.g., our_clinic_calls meddialog_audio_v2)")
    ap.add_argument("--force", action="store_true", help="Force re-download/re-export (ignore caches)")
    ap.add_argument("--rebuild-manifest", action="store_true", help="Rebuild manifests from files on disk (no re-download/convert unless needed)")
    ap.add_argument("--list", action="store_true", help="List configured datasets and exit")
    args = ap.parse_args()

    if args.list:
        print("Configured datasets:")
        for d in DATASETS:
            print(f"- {d['name']:>30}  ({d['kind']}: {d['id']})")
        sys.exit(0)

    out_root = Path(args.out)
    ensure_dir(out_root)

    selected = DATASETS
    if args.only:
        chosen = set(args.only)
        selected = [d for d in DATASETS if d["name"] in chosen]
        missing = chosen - set([d["name"] for d in selected])
        if missing:
            print(f"[!] Unknown dataset names: {', '.join(sorted(missing))}")

    produced_dirs = []
    manifests = []

    for entry in selected:
        try:
            if entry["kind"] == "kaggle":
                ds_dir = process_kaggle(entry, out_root, force=args.force, rebuild_manifest=args.rebuild_manifest)
            elif entry["kind"] == "hf":
                ds_dir = process_hf(entry, out_root, force=args.force, rebuild_manifest=args.rebuild_manifest)
            else:
                print(f"[!] Unknown kind: {entry['kind']}")
                continue
            produced_dirs.append(ds_dir)
            mpath = ds_dir / "manifest.csv"
            if mpath.exists():
                manifests.append(pd.read_csv(mpath))
        except subprocess.CalledProcessError as e:
            print(f"[!] External command failed for {entry['name']}: {e}")
        except Exception as e:
            print(f"[!] Error processing {entry['name']}: {e}")

    # Master manifest (always rebuild from per-dataset manifests we found)
    if manifests:
        master = pd.concat(manifests, ignore_index=True)
        master_csv = out_root / "master_manifest.csv"
        write_csv_atomic(master, master_csv)
        print(f"\n[✓] Master manifest: {master_csv}")
        try:
            print(master.head())
        except Exception:
            pass
    else:
        print("\n[!] No manifests found to merge.")

    print("\nDatasets available at:")
    for d in produced_dirs:
        print(f"- {d}")

if __name__ == "__main__":
    main()
