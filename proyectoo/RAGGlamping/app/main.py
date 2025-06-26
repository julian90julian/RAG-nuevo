from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_engine import responder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    pregunta: str

@app.post("/preguntar")
async def preguntar(p: Pregunta):
    respuesta = responder(p.pregunta)
    return {"respuesta": respuesta}
