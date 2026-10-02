"""Start the supported local VRM web app with one conversation worker."""
import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Yumi listening on http://127.0.0.1:{port} (local-only).")
    # One process owns the single-user conversation and its file-backed memory.
    # Bind locally as requested; this is not reachable through Replit preview.
    uvicorn.run("app:app", host="127.0.0.1", port=port, workers=1)


if __name__ == "__main__":
    main()
