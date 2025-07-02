from flask import Flask, request, jsonify
from flask_cors import CORS
from app.rag_engine import qa_chain
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"mensaje": "🟢 API de RAG corriendo correctamente"})

@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.get_json()
    pregunta = data.get("pregunta")

    if not pregunta:
        return jsonify({"error": "No se proporcionó una pregunta"}), 400

    respuesta = qa_chain.run(pregunta)
    return jsonify({"respuesta": respuesta})

# ✅ Este bloque debe estar activo en Render para que escuche el puerto correctamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
