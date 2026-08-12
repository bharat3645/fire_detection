FROM python:3.11-slim

WORKDIR /app

# System deps required by Pillow's image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core.py app.py api.py cli.py fire_detection_model.h5 ./

# 8501 = Streamlit UI (default entrypoint below), 8000 = REST API — see the
# "Run with Docker" section of the README for how to run the API image instead.
EXPOSE 8501 8000

# Checks the Streamlit UI (the default ENTRYPOINT below); not meaningful if
# the entrypoint is overridden to run the API or CLI instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# To run the REST API instead of the Streamlit UI:
#   docker run -p 8000:8000 --entrypoint uvicorn fire-detection api:app --host 0.0.0.0 --port 8000
# To run the CLI against a mounted folder of images instead:
#   docker run --entrypoint python -v /path/to/images:/data fire-detection cli.py --input /data
