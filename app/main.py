from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.rag_engine import responder

app = FastAPI()

# Habilitar CORS para permitir llamadas desde el frontend (HTML, JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringirlo a ["http://localhost:5500"] si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"mensaje": "🟢 API de RAG corriendo correctamente"}

@app.post("/preguntar")
async def preguntar(request: Request):
    data = await request.json()
    pregunta = data.get("pregunta")

    if not pregunta:
        return JSONResponse(status_code=400, content={"error": "No se proporcionó una pregunta"})

    respuesta = responder(pregunta)
    return {"respuesta": respuesta}
