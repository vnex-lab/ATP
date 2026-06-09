# VnexAI Panel

## Overview

VnexAI Panel is a Python-based application for training and running custom chatbot models. It uses a FastAPI backend in `api.py` and serves a static frontend from `frontend/dist` when available.

This repository is focused on the non-Streamlit application stack and does not require Streamlit to run the core API.

## What is included

- `api.py` - FastAPI backend and application entrypoint
- `chatbot_model.py` - custom chatbot model implementation
- `transformer_model.py` - transformer-based chatbot implementation
- `chatbot_tokenizer.py` - tokenizer and vocabulary utilities
- `frontend/` - React/Vite frontend source code
- `pyproject.toml` - Python dependency declarations

## Requirements

- Python 3.11 or higher
- Git
- Node.js and npm (only required if you want to build the frontend)

## Setup and run

### 1. Clone the repository

```bash
git clone https://github.com/vnex-lab/ATP.git
cd ATP
```

### 2. Create and activate a Python virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

Install the core dependencies from `pyproject.toml`:

```bash
python -m pip install -e .
```

If you do not need GPU support, install the CPU dependencies without CuPy:

```bash
python -m pip install fastapi matplotlib numpy pandas plotly pyarrow pydantic python-multipart scikit-learn seaborn uvicorn
```

### 4. Build the frontend (optional but recommended)

If you want the local web frontend served by the backend, build it from the `frontend` directory.

```bash
cd frontend
npm install
npm run build
cd ..
```

The built frontend files are placed in `frontend/dist`. The backend serves those files automatically if the directory exists.

### 5. Start the backend server

Run the FastAPI app with the internal Uvicorn server:

```bash
python api.py
```

The server starts on port `5000` by default.

### 6. Open the app in your browser

If the frontend build exists, open:

```text
http://localhost:5000
```

If you only use the API, the backend is available at `http://127.0.0.1:5000`.

## Quick run commands

Start the app directly:

```bash
python api.py
```

Or use Uvicorn with live reload:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 5000
```

## Common troubleshooting

### Port is already in use

```bash
python -c "from api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=5001)"
```

### GPU is not available

The code falls back to CPU automatically if CuPy is not usable. If you want GPU support, install the correct CuPy package for your CUDA version.

### Frontend does not load

Make sure the frontend build exists:

```bash
cd frontend
npm install
npm run build
```

## File layout

- `api.py` - application backend and static file hosting
- `chatbot_model.py` - RNN chatbot model code
- `transformer_model.py` - transformer chatbot code
- `chatbot_tokenizer.py` - tokenizer and vocabulary utilities
- `frontend/` - React/Vite source and build output
- `pyproject.toml` - Python dependency file

## API endpoints

The backend exposes endpoints for data upload, model setup, and status checks. Key routes include:

- `GET /api/status`
- `POST /api/data/upload-file`
- `POST /api/data/upload-text`
- `POST /api/data/load-builtin`
- `GET /api/data/info`

Use the frontend or your own HTTP client to interact with these endpoints.

## Notes

- This repository is configured for the FastAPI-based application.
- The root `api.py` file is the main entrypoint for running the service.
- Building the frontend is optional if you only need the API.
