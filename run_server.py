import uvicorn
uvicorn.run("agent.main:app", host="127.0.0.1", port=8000, log_level="info")
