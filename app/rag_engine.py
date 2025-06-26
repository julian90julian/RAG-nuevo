from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import os
from openai import OpenAIError
from dotenv import load_dotenv

# --- Configuración Inicial y Carga de Componentes (solo una vez al iniciar la app) ---

# 🔐 Cargar clave API desde .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ Error: No se encontró la variable OPENAI_API_KEY en el archivo .env")
    # Para entornos de producción como Railway, es mejor que esto cause un fallo
    # en el inicio si la clave no está, ya que la aplicación no funcionaría.
    exit(1) # Salir con un código de error

# Inicializar Flask
app = Flask(__name__)

# --- Inicialización de OpenAI y Langchain ---
chat = None
qa_chain = None
vectorstore = None

try:
    print("🔌 Verificando conexión con OpenAI y configurando Langchain...")
    chat = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7) # Añadir temperature es buena práctica
    
    # 📄 Leer documento
    ruta_doc = "data/documentos.txt" # Ajusta la ruta si es necesario
    if not os.path.exists(ruta_doc):
        print(f"❌ Error: Archivo no encontrado: {ruta_doc}")
        exit(1)

    with open(ruta_doc, encoding="utf-8") as f:
        texto = f.read()

    if not texto.strip():
        print("❌ Error: El archivo está vacío. Agrega contenido a 'documentos.txt'.")
        exit(1)

    print(f"📄 Documento cargado ({len(texto)} caracteres)")

    # 🧩 Fragmentar texto
    chunks = [Document(page_content=texto[i:i+500]) for i in range(0, len(texto), 500)]
    print(f"✂️ Fragmentado en {len(chunks)} partes")

    # 🔍 Embeddings y FAISS
    print("📦 Generando embeddings y vectorstore...")
    # Verificar si el índice ya existe para cargarlo en vez de crearlo
    if os.path.exists("faiss_index"):
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vectorstore FAISS cargado desde disco.")
    else:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("faiss_index")
        print("✅ Vectorstore FAISS creado y guardado en disco.")

    # 🔁 Crear cadena de respuesta
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    print("✅ Cadena de QA de Langchain configurada.")

except OpenAIError as e:
    print(f"❌ Error al conectar o configurar OpenAI: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error general durante la inicialización de Langchain: {e}")
    exit(1)

# --- Rutas de la API Flask ---

# Ruta raíz para verificar que el servidor está funcionando
@app.route("/")
def home():
    if qa_chain:
        return "¡Servidor Flask para Glamping con IA funcionando correctamente!"
    else:
        return "Servidor Flask funcionando, pero la IA no se inicializó correctamente.", 500

# Ruta para responder preguntas
@app.route("/ask", methods=["POST"])
def ask_question():
    if not qa_chain:
        return jsonify({"error": "La cadena de IA no está inicializada."}), 500

    data = request.get_json()
    pregunta = data.get("question")

    if not pregunta:
        return jsonify({"error": "Por favor, proporciona una pregunta en el cuerpo de la solicitud (campo 'question')."}), 400

    try:
        print(f"🤖 Recibida pregunta de usuario: {pregunta}")
        respuesta = qa_chain.run(pregunta)
        print("✅ Respuesta generada y enviada.")
        return jsonify({"question": pregunta, "answer": respuesta})
    except Exception as e:
        print(f"❌ Error al procesar la pregunta: {e}")
        return jsonify({"error": f"Error al procesar la pregunta: {str(e)}"}), 500

# --- Iniciar el servidor Flask (solo para desarrollo local) ---
# En Railway, Gunicorn iniciará la aplicación por ti, esta sección se ignora.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Iniciando servidor Flask en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True) # debug=True solo para desarrollo