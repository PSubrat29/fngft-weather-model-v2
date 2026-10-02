# Cloud deployment design

## Reference topology

```text
real-time object storage / data feed
            ↓
      ingestion job
            ↓
   validation + QC job        (python -m fngft inspect)
            ↓
 standardized training dataset
            ↓
       GPU training           (python -m fngft train)
            ↓
 checkpoint + metadata registry   (evaluate report stored next to the checkpoint)
            ↓
 container image              (Dockerfile)
            ↓
 managed inference service    (uvicorn fngft.api:app)
            ↓
 forecast API + dashboard
```

## Container

```bash
docker build -t fngft .
docker run -p 8080:8080 \
  -v /srv/fngft/artifacts:/app/artifacts \
  -v /srv/fngft/data:/app/data \
  -v /srv/fngft/configs:/app/deploy \
  -e MODEL_PATH=/app/artifacts/fngft_real.pt \
  -e FNGFT_CONFIG=/app/deploy/my_data.yaml \
  fngft
```

The image uses CPU PyTorch. The service starts even if the checkpoint is missing and reports the
problem on `/health`; `POST /reload` picks up a newly mounted checkpoint.

## Real-time forecast mode

A scheduler can invoke, whenever a new data file arrives:

```bash
python -m fngft forecast-latest --config CONFIG --checkpoint CKPT --steps 6 --output out/forecast_$(date +%Y%m%d%H).nc
```

or call `GET /forecast-latest` on the running service.

## Separation of duties

- **Data plane** — stores raw and normalized datasets.
- **Training plane** — creates immutable model artifacts.
- **Inference plane** — serves one selected approved model version.
- **Governance plane** — records provenance, validation metrics and release status.

## Security

The HTTP API never accepts filesystem paths. Dataset locations are deployment configuration, not
request parameters. Add authentication and rate limiting before exposing the service publicly.

## Scaling

Inference can be horizontally scaled because the checkpoint is immutable and each request is
independent. Data ingestion and training stay separate from the inference service.
