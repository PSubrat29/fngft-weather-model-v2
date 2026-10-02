# Cloud deployment design

## Reference topology

```text
real-time object storage / data feed
            ↓
      ingestion job
            ↓
   validation + QC job
            ↓
 standardized training dataset
            ↓
       GPU training
            ↓
 checkpoint + metadata registry
            ↓
 container image
            ↓
 managed inference service
            ↓
 forecast API
```

## Real-time forecast mode

A scheduler can invoke:

```bash
python -m fngft.cli forecast-latest ...
```

whenever a new data file arrives.

## Separation of duties

### Data plane
Stores raw and normalized datasets.

### Training plane
Creates immutable model artifacts.

### Inference plane
Serves one selected approved model version.

### Governance plane
Records provenance, validation metrics and release status.

## Security

The HTTP API should never accept arbitrary server filesystem paths. Dataset locations are deployment configuration, not request parameters.

## Scaling

Inference can be horizontally scaled because the model checkpoint is immutable and each request is independent. Data ingestion/training should remain separate from the inference service.
