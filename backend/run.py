# pyrefly: ignore [missing-import]
import uvicorn
import os
import sys

# Ensure backend directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
