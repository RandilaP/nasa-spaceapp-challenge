FROM continuumio/miniconda3:latest

WORKDIR /app

# Create a conda env and install heavy dependencies from conda-forge
RUN conda update -n base -c defaults conda -y \
    && conda create -n airq python=3.11 -y \
    && /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate airq && conda install -c conda-forge -y \
       netcdf4 h5py numpy pandas scikit-learn joblib pyyaml requests fastapi uvicorn pydantic" \
    && conda clean -afy

# Ensure the conda env is on PATH
ENV PATH /opt/conda/envs/airq/bin:$PATH

# Copy project files
COPY . /app

# Install any remaining pip-only requirements (if present)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt || true

# Create data dirs
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models \
    && chmod -R a+rwx /app/data

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
