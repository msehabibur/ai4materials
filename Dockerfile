# =========================
# Streamlit on Python 3.10
# =========================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gfortran git curl \
    libgomp1 libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 \
    libfreetype6 libjpeg62-turbo libpng16-16 \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user and app directory
ARG APP_USER=appuser
RUN useradd -ms /bin/bash ${APP_USER}

# Set up the app directory and permissions
WORKDIR /app
# Create the .streamlit directory and set its permissions first
RUN mkdir -p /app/.streamlit
# Now change the ownership of the entire /app directory to the new user
RUN chown -R ${APP_USER}:${APP_USER} /app

# Switch to the non-root user
USER ${APP_USER}

# Copy requirements first for better cache
COPY --chown=${APP_USER}:${APP_USER} requirements.txt /app/requirements.txt

# Install pip + deps
RUN python -m pip install --upgrade pip wheel \
 && pip install -r requirements.txt

# Copy app code
COPY --chown=${APP_USER}:${APP_USER} . /app

# Streamlit config (now runs as appuser, who owns /app/.streamlit)
RUN printf "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = true\nport = 8080\naddress = \"0.0.0.0\"\n\n[browser]\ngatherUsageStats = false\n" \
    > /app/.streamlit/config.toml

# Cloud Run/AWS expect $PORT
ENV PORT=8080
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT}/ || exit 1

# Run Streamlit
CMD ["bash", "-lc", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
