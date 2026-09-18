FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CAUSALRAG_API_HOST=0.0.0.0
ENV CAUSALRAG_API_PORT=8765
ENV CAUSALRAG_SURFACE=probe

COPY pyproject.toml setup.py setup.cfg MANIFEST.in README.MD LICENSE ./
COPY causalrag ./causalrag

RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e '.[api]'

EXPOSE 8765

CMD ["python", "-m", "causalrag.cli", "probe", "--host", "0.0.0.0", "--port", "8765"]
