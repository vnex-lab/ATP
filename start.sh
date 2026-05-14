#!/bin/bash
set -e

echo "==> Building frontend..."
cd frontend
npm install --prefer-offline 2>&1 | tail -5
npm run build 2>&1
cd ..

echo "==> Starting FastAPI server..."
exec uvicorn api:app --host 0.0.0.0 --port 5000
