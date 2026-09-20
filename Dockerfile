FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV BRANCHPOINT_API_HOST=0.0.0.0
ENV BRANCHPOINT_API_PORT=8765
ENV BRANCHPOINT_SURFACE=probe

COPY pyproject.toml setup.py setup.cfg MANIFEST.in README.MD LICENSE ./
COPY branchpoint ./branchpoint

RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e '.[api]'

EXPOSE 8765

CMD ["python", "-m", "branchpoint.cli", "probe", "--host", "0.0.0.0", "--port", "8765"]
