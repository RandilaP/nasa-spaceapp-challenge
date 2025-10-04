FROM python:3.11-slim

# Install system dependencies required for netCDF4, h5py, and other scientific libs
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
         build-essential \
         ca-certificates \
         libhdf5-dev \
         libnetcdf-dev \
         netcdf-bin \
         git \
         wget \
     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement file first to leverage layer caching
COPY requirements.txt ./

# Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . /app

# Create data dirs with permissive permissions for runtime writes
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models \
    && chmod -R a+rwx /app/data

# Expose port used by uvicorn
EXPOSE 8000

# Default command: run uvicorn serving the FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
