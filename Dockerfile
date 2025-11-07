# Stage 1: build frontend (Vite)
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# install deps and build
COPY frontend/package.json frontend/package-lock.json* ./ 
RUN npm install --no-audit --no-fund
COPY frontend/ .
RUN npm run build

# Stage 2: python runtime with backend and static frontend
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy backend and builtin frontend build
COPY backend/ ./backend
COPY --from=frontend-builder /app/frontend/dist ./backend/frontend/dist

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
ENV PORT=8080
EXPOSE 8080

# Start the FastAPI app (serves static frontend from backend/frontend/dist)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
