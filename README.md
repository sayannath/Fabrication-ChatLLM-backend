# Fabrication ChatLLM Backend

## Start the server

```bash
export LOG_LEVEL=INFO
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Output

```
Uvicorn running on http://127.0.0.1:8000
```

[API Documentation](http://127.0.0.1:8000/docs)