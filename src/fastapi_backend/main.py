from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_backend.routers import logs, users

app = FastAPI(title="Logchain Recon API")

# Allow frontend (SvelteKit) to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs.router)
app.include_router(users.router)
# app.include_router(graphs.router)
# app.include_router(reports.router)

@app.get("/")
def root():
    return {"message": "Logchain Recon backend is running"}
