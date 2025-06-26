from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import openai
import os
from openai import OpenAIError
from dotenv import load_dotenv

# 🔐 Cargar clave API desde .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

if not api_key:
    raise ValueError("❌ No se encontró la variable OPENAI_API_KEY en el archivo .env")

openai.api_key = api_key

# 🧪 Paso 1: Verificar conexión con OpenAI
print("🔌 Verificando conexión con OpenAI...")
try:
    chat = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
    prueba = chat.invoke("Hola, ¿puedes responder?")
    print("✅ OpenAI está funcionando:", prueba.content)
except OpenAIError as e:
    print("❌ Error al conectarse con OpenAI:", e)
    exit()