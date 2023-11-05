from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Creating our app
app = FastAPI()

# Define CORS middleware
origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

# Implement default API endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI setup successfully"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
