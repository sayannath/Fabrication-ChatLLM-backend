# Fabrication ChatLLM Backend

## Start the server

```bash
export LOG_LEVEL=INFO
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
# Optional overrides
export FABRICATION_DATASET_PATH="data/LLM-Dataset - Fabrication.csv"
export WEAVE_PROJECT="fabrication-rag"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Output

```
Uvicorn running on http://127.0.0.1:8000
```

[API Documentation](http://127.0.0.1:8000/docs)

## Retrieval-Augmented Generation

- The backend loads the `data/LLM-Dataset - Fabrication.csv` file into an in-memory corpus and uses a lightweight BM25 retriever to surface the most relevant fabrication research snippets for each query.
- DSPy composes the retrieval results (`contexts`) with the configured language model to generate grounded answers.
- Responses now include the supporting contexts and scored metadata for each cited paper to aid downstream UX.

## Weave Instrumentation

- If the optional [W&B Weave](https://wandb.ai/site/weave) dependency is installed, the pipeline logs retrieval and generation events.
- Configure the project namespace via `WEAVE_PROJECT`; additional authentication (e.g., `WANDB_API_KEY`) can be provided through the environment as needed.
- The backend degrades gracefully when Weave is not available—no configuration changes are required for local development without telemetry.
