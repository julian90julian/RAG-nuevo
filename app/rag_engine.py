from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import openai
import os
from openai import OpenAIError
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


# 🔐 Cargar clave API desde .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("🔐 Clave API cargada correctamente.")  #se pone este print(api_key)

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

# 📄 Paso 2: Leer documento
ruta_doc = "app/data/documentos.txt"
if not os.path.exists(ruta_doc):
    print(f"❌ Archivo no encontrado: {ruta_doc}")
    exit()

with open(ruta_doc, encoding="utf-8") as f:
    texto = f.read()

if not texto.strip():
    print("❌ El archivo está vacío. Agrega contenido.")
    exit()

print(f"📄 Documento cargado ({len(texto)} caracteres)")

# 🧩 Paso 3: Fragmentar texto
chunks = [Document(page_content=texto[i:i+500]) for i in range(0, len(texto), 500)]
print(f"✂️ Fragmentado en {len(chunks)} partes")

# 🔍 Paso 4: Embeddings y FAISS
print("📦 Generando embeddings y vectorstore...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("faiss_index")

# 🧪 Paso 5: Ver fragmentos que se buscan
consulta = "¿Qué servicios ofrece el glamping?"
print(f"🔍 Buscando documentos relevantes para: {consulta}")
docs = vectorstore.similarity_search(consulta, k=3)
for i, doc in enumerate(docs):
    print(f"📄 Documento {i+1}: {doc.page_content[:200]}...\n")

# 🔁 Paso 6: Crear cadena de respuesta
qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 🎯 Paso 7: Hacer la pregunta
def responder(pregunta: str) -> str:
    try:
        print(f"🤖 Enviando pregunta: {pregunta}")
        respuesta = qa_chain.run(pregunta)
        print("✅ Respuesta recibida.")
        return respuesta
    except Exception as e:
        return f"[Error] {str(e)}"

# ▶️ Ejecutar
if __name__ == "__main__":
    resultado = responder(consulta)
    print("\n💬 Respuesta final:\n", resultado)
