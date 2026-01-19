# web_server.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit import api
from dotenv import load_dotenv
import uvicorn
from pypdf import PdfReader

load_dotenv()

app = FastAPI()

# Path where we save the uploaded doc for the MCP server to read
KNOWLEDGE_FILE = "shared_knowledge.txt"

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

# 2. File Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a file, extracts text (handles PDF and text files), 
    and saves it for the MCP server.
    """
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        text_content = ""
        
        if file_extension == ".pdf":
            # Extract text from PDF
            pdf_reader = PdfReader(file.file)
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
        else:
            # For text files, read directly
            content = await file.read()
            text_content = content.decode("utf-8", errors="ignore")
        
        # Save the extracted text content
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as buffer:
            buffer.write(text_content)
            
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 3. Serve index.html explicitly at root
@app.get("/")
async def read_index():
    return FileResponse('index.html')

# 4. Mount other static files
app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    # Ensure previous knowledge is cleared on restart (optional)
    if os.path.exists(KNOWLEDGE_FILE):
        os.remove(KNOWLEDGE_FILE)
        
    print(f"Web server running. Shared knowledge file: {os.path.abspath(KNOWLEDGE_FILE)}")
    uvicorn.run(app, host="0.0.0.0", port=3000)