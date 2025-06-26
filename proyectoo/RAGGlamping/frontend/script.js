
async function hacerPregunta() {
    const pregunta = document.getElementById("pregunta").value;
    const res = await fetch("http://127.0.0.1:8000/preguntar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta })
    });
    const data = await res.json();
    document.getElementById("respuesta").innerText = data.respuesta;
}
