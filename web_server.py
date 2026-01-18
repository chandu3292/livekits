# web_server.py
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit import api
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# 1. Define the Token Endpoint
@app.get("/token")
def get_token():
    at = api.AccessToken(
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    ).with_identity("web-user").with_grants(
        api.VideoGrants(room_join=True, room="default")
    )

    return {
        "token": at.to_jwt(),
        "url": os.environ["LIVEKIT_URL"],
    }

# 2. Serve index.html explicitly at root
@app.get("/")
async def read_index():
    return FileResponse('index.html')

# 3. Mount other static files (if you have CSS/JS files)
app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    # Run on port 3000 to match your current frontend setup
    uvicorn.run(app, host="0.0.0.0", port=3000)