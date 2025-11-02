# Medical Speech Analysis Results Browser

A Dockerized web application for browsing and analyzing medical speech datasets stored in Pixeltable.

## Features

- 🎤 **Audio Dataset Browser**: Browse medical speech transcriptions with filtering and search
- 📊 **Analytics Dashboard**: Visualize dataset statistics and transcription patterns  
- 🎵 **Audio Player**: Play and analyze audio files (planned)
- 🐳 **Dockerized**: Easy deployment and sharing
- 💾 **Local Caching**: Preloads dataset for faster access
- 🔐 **Secure**: Uses environment variables for API keys

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Access to the Pixeltable dataset: `pxt://speech-to-text-analytics/hani89_asr_dataset`

### Setup

1. **Copy environment file**:
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` file** with your Pixeltable API key:
   ```bash
   PIXELTABLE_API_KEY=your_actual_api_key_here
   PIXELTABLE_DATASET_URL=pxt://speech-to-text-analytics:main/hani89_asr_dataset
   ```

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   Open your browser to `http://localhost:8050`

### Manual Setup (without Docker)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export PIXELTABLE_API_KEY=your_api_key_here
   export PIXELTABLE_DATASET_URL=pxt://speech-to-text-analytics:main/hani89_asr_dataset
   ```

3. **Preload data**:
   ```bash
   python preload_data.py
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

## Usage

### Data Table Tab
- Browse all transcriptions in a searchable, filterable table
- Filter by dataset split (train/test/validation)
- Search within transcriptions
- View Whisper model results

### Analytics Tab
- View transcription length distributions
- Analyze dataset split proportions
- Explore data patterns

### Audio Player Tab
- Play audio files (when available)
- Visualize waveforms
- Compare transcriptions

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PIXELTABLE_API_KEY` | Your Pixeltable API key | Required |
| `PIXELTABLE_DATASET_URL` | Dataset URL to load | `pxt://speech-to-text-analytics/hani89_asr_dataset` |
| `CACHE_DIR` | Local cache directory | `./cache` |
| `DATA_DIR` | Data directory | `./data` |

### Caching

The application automatically caches the dataset locally for faster access. The cache is stored in the `cache/` directory and includes:

- `dataset_cache.pkl`: Cached pandas DataFrame
- `dataset_metadata.json`: Dataset metadata and timestamps

To refresh the cache, delete the cache files and restart the application.

## Development

### Project Structure

```
results_browser/
├── app.py                 # Main Dash application
├── preload_data.py        # Data preloading script
├── startup.sh            # Docker startup script
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
├── env.example          # Environment variables template
├── cache/               # Local data cache
└── data/               # Additional data files
```

### Adding Features

1. **New Data Sources**: Modify `preload_data.py`
2. **UI Components**: Update `app.py` layout and callbacks
3. **Analytics**: Add new visualizations in the analytics tab
4. **Audio Features**: Implement audio player functionality

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure `.env` file exists with correct `PIXELTABLE_API_KEY`
2. **Dataset Not Loading**: Check network connection and API key validity
3. **Cache Issues**: Delete `cache/` directory to force refresh
4. **Port Conflicts**: Change port in `docker-compose.yml` if 8050 is occupied

### Logs

View application logs:
```bash
docker-compose logs -f results-browser
```

## License

This project is part of the Medical Speech Analysis research project.
