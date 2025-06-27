from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from openai import OpenAIError

# 🔐 Cargar clave API desde .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ Error: No se encontró la variable OPENAI_API_KEY en el archivo .env")
    exit(1)

chat = None
qa_chain = None
vectorstore = None

try:
    print("🔌 Verificando conexión con OpenAI y configurando Langchain...")
    chat = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

    ruta_doc = "data/documentos.txt"
    if not os.path.exists(ruta_doc):
        print(f"❌ Error: Archivo no encontrado: {ruta_doc}")
        exit(1)

    with open(ruta_doc, encoding="utf-8") as f:
        texto = f.read()

    if not texto.strip():
        print("❌ Error: El archivo está vacío.")
        exit(1)

    print(f"📄 Documento cargado ({len(texto)} caracteres)")

    chunks = [Document(page_content=texto[i:i+500]) for i in range(0, len(texto), 500)]
    print(f"✂️ Fragmentado en {len(chunks)} partes")

    if os.path.exists("faiss_index"):
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vectorstore FAISS cargado desde disco.")
    else:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("faiss_index")
        print("✅ Vectorstore FAISS creado y guardado en disco.")

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
