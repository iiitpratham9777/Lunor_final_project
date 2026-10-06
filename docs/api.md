# API Reference

Base: `/api/v1`

| Method | Path | Description |
|--------|------|-------------|
| POST | /investigations | Create investigation |
| POST | /investigations/{id}/images | Upload one or more images |
| POST | /investigations/{id}/run | Execute PEV agent loop |
| GET | /investigations/{id} | Status + decision |
| GET | /investigations/{id}/trace | Full agent trace + tool calls |
| GET | /investigations/{id}/evidence | Observations, evidence, confidence |
| GET | /investigations/{id}/report | Final report + explanation |
| GET | /investigations | List investigations |
| GET | /health | Health check |

OpenAPI UI: `http://localhost:8000/docs`
