# FastRay Container Setup - Path Fixes Applied

## ✅ Changes Made

Fixed the bind mounts in `run_fastray.sh` to point to the correct directory structure:

### Before (pointing to dagutman):
```bash
-v /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats:/data \
-v /scr/dagutman/devel/medSpeech_Analysis/medSpeechAnalysis_hf_ray:/data_medspeech \
```

### After (pointing to dgutman with new structure):
```bash
-v /scr/dgutman/devel/medSpeech_Analysis/train_data:/data/train_data \
-v /scr/dgutman/devel/medSpeech_Analysis/test_data:/data/test_data \
```

## Updated Files

1. ✅ **`run_fastray.sh`**: Fixed paths to `/scr/dgutman/` and mount structure
2. ✅ **`transcribe_helpers.py`**: Updated path mapping logic to handle `train_data/` and `test_data/` subdirectories

## Path Mapping

Your audio files are stored as:
- Host: `/scr/dgutman/devel/medSpeech_Analysis/train_data/train_sample_0.wav`
- Container: `/data/train_data/train_sample_0.wav`

The `transcribe_helpers.py` now correctly maps:
- `train_data/train_sample_0.wav` → `/data/train_data/train_sample_0.wav`
- `test_data/test_sample_0.wav` → `/data/test_data/test_sample_0.wav`

## To Start FastRay Container

```bash
cd /scr/dgutman/devel/medSpeech_Analysis/services/fastRay

# Start with default config (or set NUM_REPLICAS in .env)
./run_fastray.sh

# Or with specific replica count
NUM_REPLICAS=40 NUM_GPUS_PER_REPLICA=0.1 ./run_fastray.sh
```

## To Run Transcriptions

```bash
cd /scr/dgutman/devel/medSpeech_Analysis

# Set lockfile to avoid permission issues
export PIXELTABLE_LOCKFILE=/tmp/.lockfile.$USER

# Run transcriptions
python transcribe_via_raycontainer.py
```

## Verify Container Mounts

```bash
# Check container is running
docker ps | grep fastray

# Verify mounts inside container
docker exec fastray-container ls -la /data/
# Should show: train_data/ and test_data/ directories

docker exec fastray-container ls /data/train_data/ | head -5
# Should show: train_sample_0.wav, train_sample_1.wav, etc.
```

## Notes

- The container will have access to all 6,661 audio files
- API endpoint: `http://localhost:8000`
- Ray Dashboard: `http://localhost:8265`
- Model cache persists at: `/scr/dgutman/.../services/fastRay/model_cache`
