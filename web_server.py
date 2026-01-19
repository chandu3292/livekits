import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit import api
from dotenv import load_dotenv
import uvicorn
from pypdf import PdfReader

load_dotenv()

app = FastAPI()

KNOWLEDGE_FILE = "shared_knowledge.txt"
RAG_SERVER_URL = "http://localhost:8000/trigger-update"

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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a file, extracts text, saves it, and TRIGGERS the RAG server.
    """
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        text_content = ""
        
        if file_extension == ".pdf":
            pdf_reader = PdfReader(file.file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
        else:
            content = await file.read()
            text_content = content.decode("utf-8", errors="ignore")
        
        # 1. Save Text to Disk
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as buffer:
            buffer.write(text_content)
            
        print(f"✅ File saved: {file.filename} ({len(text_content)} chars)")

        # 2. Trigger RAG Server Update
        print("🚀 Triggering RAG Index Update...")
        try:
            response = requests.post(RAG_SERVER_URL, timeout=10)
            if response.status_code == 200:
                print("✅ RAG Server Updated Successfully")
            else:
                print(f"⚠️ RAG Server returned status: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to contact RAG server: {e}")
            print("Is server.py running?")

        return {"status": "success", "filename": file.filename}

    except Exception as e:
        print(f"Error handling upload: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def read_index():
    return FileResponse('index.html')

app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)