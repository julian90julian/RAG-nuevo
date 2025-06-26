from flask import Flask, request, jsonify
from flask_cors import CORS
from app.rag_engine import qa_chain

app = Flask(__name__)

# Habilitar CORS para permitir llamadas desde el frontend
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

    # Aquí cambiamos la llamada a responder por qa_chain
    respuesta = qa_chain.run(pregunta)
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
