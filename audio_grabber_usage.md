# 1) deps
pip install datasets soundfile pandas kaggle pydub tqdm
# + ffmpeg installed (apt/brew/etc.)

# 2) Kaggle API creds (once)
# put kaggle.json at ~/.kaggle/kaggle.json and chmod 600

# 3) run – downloads & exports everything, safely resumable
python med_audio_grabber_resumable.py

# list configured datasets
python med_audio_grabber_resumable.py --list

# run only some datasets (safe to repeat)
python med_audio_grabber_resumable.py --only our_clinic_calls meddialog_audio_v2

# rebuild manifests from what’s on disk (no re-download unless needed)
python med_audio_grabber_resumable.py --rebuild-manifest

# force a clean re-fetch/re-export
python med_audio_grabber_resumable.py --force
