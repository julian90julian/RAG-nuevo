<<<<<<< HEAD
# RAGGlamping
=======
# Sistema de Preguntas y Respuestas con RAG

Este proyecto implementa un sistema de preguntas y respuestas utilizando RAG (Retrieval-Augmented Generation) con FastAPI y OpenAI.

## Requisitos

- Python 3.8 o superior
- API Key de OpenAI

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv .venv
```

2. Activar el entorno virtual:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar API Key:
Editar el archivo `app/rag_engine.py` y reemplazar la API key de OpenAI con tu propia key.

## Ejecución

1. Iniciar el servidor:
```bash
python app/main.py
```

2. El servidor estará disponible en `http://localhost:8000`

3. Puedes hacer preguntas usando el endpoint `/preguntar` con una petición POST:
```json
{
    "pregunta": "Tu pregunta aquí"
}
```

## Estructura del Proyecto

- `app/`: Directorio principal de la aplicación
  - `main.py`: Configuración de FastAPI y endpoints
  - `rag_engine.py`: Implementación del motor RAG
  - `data/`: Directorio con los documentos para el RAG 
>>>>>>> 97ede40 (proyecto RAG para Railway)
