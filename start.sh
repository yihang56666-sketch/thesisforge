#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "========================================"
echo "  毕设工坊 ThesisForge  http://127.0.0.1:8765"
echo "========================================"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
