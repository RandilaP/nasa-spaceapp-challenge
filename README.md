# Air Quality Forecasting — Render deployment guide

This project provides an ML pipeline and a FastAPI backend (`src.api:app`) that serves current AQI, forecasts and metrics. This README explains how to deploy the service to Render as a web service (Docker or Python service) and how to verify endpoints.

## Recommended approach

1. Use the included `Dockerfile` (recommended) to ensure system libraries for `netCDF4`/`h5py` are available.
2. Alternatively use Render's Python service, but add the required apt packages via a build command (less reliable).

## Files added for deployment

- `Dockerfile` — builds a slim Python image, installs system libs, pip dependencies, and runs uvicorn.
- `.dockerignore` — keep build context small and avoid uploading data and tests.
- `Procfile` — convenience for Render's Python service type (runs uvicorn with $PORT).

## Environment variables

- `MODEL_PATH` — optional path to a pre-trained model (default `./data/models/best_model.pkl`).
- `LATEST_DATA_PATH` — optional path to processed data CSV (default `./data/processed/processed_data.csv`).

If you use Render's dashboard, set these under the service's Environment tab.

## Deploying on Render (Docker)

1. Create a new Web Service on Render and choose "Docker" as the environment.
2. Connect your GitHub repo and select the branch (e.g. `main`).
3. Render will build the Dockerfile automatically. Build logs show pip and apt steps.

Tips for build failures:
- If `netCDF4` or `h5py` fails to build, ensure the Dockerfile installs `libhdf5-dev` and `libnetcdf-dev` (already included).
- If you see unresolved system deps, add them to the `apt-get install` line in the Dockerfile.

## Deploying on Render (Python service)

1. Create a new Web Service and choose "Python".
2. Use the `requirements.txt` provided.
3. In the Build Command, install system deps before pip, for example:

```bash
apt-get update && apt-get install -y libhdf5-dev libnetcdf-dev
pip install -r requirements.txt
```

4. Set the Start Command to:

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

Note: Render's managed Python service may not give you enough control to add system packages. When in doubt, use the Docker service.

## Quick local test (before pushing)

Build and run the Docker image locally:

```bash
docker build -t airq-api:local .
docker run -p 8000:8000 --env MODEL_PATH=./data/models/best_model.pkl --env LATEST_DATA_PATH=./data/processed/processed_data.csv airq-api:local
```

Then open http://localhost:8000/docs

## How to check endpoints after deployment

Use `curl` (replace HOST with your Render URL, e.g. `https://your-service.onrender.com`):

Health check

```bash
curl -sS $HOST/ | jq
```

Current AQI

```bash
curl -sS $HOST/current | jq
```

Forecast (POST)

```bash
curl -sS -X POST $HOST/forecast -H "Content-Type: application/json" -d '{"hours":24}' | jq
```

Metrics

```bash
curl -sS $HOST/metrics | jq
```

If endpoints return 503, confirm `MODEL_PATH` and `LATEST_DATA_PATH` files exist and are readable by the service.

## Common errors and fixes

- Build errors for `netCDF4`/`h5py`: ensure system libs installed in Dockerfile (`libhdf5-dev`, `libnetcdf-dev`) and upgrade pip before install.
- `ModuleNotFoundError` / missing packages: verify `requirements.txt` is up to date and Render installed dependencies during build.
- 503 responses from API: check service logs for errors during startup — missing `config.yaml`, missing model, or exceptions in `startup_event`.

## Next steps & automation

- (Optional) Add a small script to push a pre-trained model and processed CSV to the repo or a storage bucket and set `MODEL_PATH`/`LATEST_DATA_PATH` accordingly.

---

If you'd like, I can:

- Add a small health-check shell script to call endpoints repeatedly and assert they return expected status codes.
- Add a lightweight GitHub Action to build the Docker image and push to Render via `render-cli` on push.
