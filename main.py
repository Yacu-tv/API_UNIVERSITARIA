from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from collections import Counter
from dotenv import load_dotenv
import spacy
import pandas as pd
import groq
import os
import io
import json

# --------------------------------------------------
# CONFIGURACIÓN INICIAL
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("No se encontró GROQ_API_KEY en el archivo .env")

cliente_groq = groq.Groq(api_key=GROQ_API_KEY)

# Modelo spaCy en español
nlp = spacy.load("es_core_news_sm")

app = FastAPI(
    title="API de Análisis Cualitativo con IA",
    description="Plataforma de análisis cualitativo para cualquier tipo de encuesta o tema",
    version="2.1.0"
)

# --------------------------------------------------
# CORS — permite ser llamada desde cualquier lugar
# frontend, apps móviles, Postman, Google Sheets, etc.
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# STOPWORDS
# --------------------------------------------------

STOPWORDS = nlp.Defaults.stop_words | {
    "siento", "estoy", "existe", "mucha", "mucho", "tantas",
    "ultimamente", "tener", "tengo", "hacer", "puede", "pueden",
    "persona", "personas", "cosas", "tema", "temas", "bien", "mal"
}

# --------------------------------------------------
# MODELOS DE ENTRADA
# --------------------------------------------------

class RespuestaInput(BaseModel):
    texto: str

class LoteInput(BaseModel):
    respuestas: List[str]
    contexto: Optional[str] = "análisis cualitativo"

class GoogleSheetsInput(BaseModel):
    datos: List[dict]
    columna_respuestas: str
    contexto: Optional[str] = "análisis cualitativo"


# --------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------

def extraer_palabras_clave(texto: str) -> List[str]:
    """Extrae palabras clave usando spaCy."""
    try:
        doc = nlp(texto.lower())
        palabras_clave = []

        for token in doc:
            if (
                token.text not in STOPWORDS
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 3
                and token.pos_ in ["NOUN", "ADJ", "VERB"]
            ):
                palabras_clave.append(token.lemma_)

        conteo = Counter(palabras_clave)
        return [palabra for palabra, _ in conteo.most_common(5)]

    except Exception:
        return []


def analizar_con_groq(texto: str, contexto: str = "análisis cualitativo") -> dict:
    """
    Analiza una respuesta abierta usando LLaMA 3 via Groq.
    Devuelve emocion, nivel_riesgo e interpretacion.
    """
    prompt = f"""Eres un sistema experto en análisis cualitativo de texto.

Analiza la siguiente respuesta en el contexto de: {contexto}

Responde ÚNICAMENTE con este formato JSON, sin texto adicional:

{{
  "emocion": "positiva" | "negativa" | "neutra",
  "nivel_riesgo": "alto" | "medio" | "bajo",
  "confianza": número entre 0.0 y 1.0,
  "interpretacion": "breve explicación en español de máximo 2 oraciones"
}}

Criterios de nivel de riesgo:
- alto: menciona situaciones graves, crisis, problemas severos, urgencia
- medio: menciona dificultades, insatisfacción, problemas moderados
- bajo: respuesta positiva, neutra o sin señales de alerta

Respuesta a analizar:
"{texto}"

Responde solo con el JSON:"""

    try:
        respuesta = cliente_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        contenido = respuesta.choices[0].message.content.strip()
        contenido = contenido.replace("```json", "").replace("```", "").strip()

        resultado = json.loads(contenido)

        return {
            "emocion_detectada": resultado.get("emocion", "neutra"),
            "nivel_riesgo": resultado.get("nivel_riesgo", "bajo"),
            "confianza_modelo": round(float(resultado.get("confianza", 0.5)), 2),
            "interpretacion": resultado.get("interpretacion", "Sin interpretación disponible.")
        }

    except Exception as e:
        return {
            "emocion_detectada": "neutra",
            "nivel_riesgo": "bajo",
            "confianza_modelo": 0.0,
            "interpretacion": f"Error al analizar: {str(e)}"
        }


def generar_resumen_lote(resultados: List[dict], contexto: str) -> dict:
    """Genera conclusión y recomendación general del lote."""
    positivas = sum(1 for r in resultados if r["emocion_detectada"] == "positiva")
    negativas = sum(1 for r in resultados if r["emocion_detectada"] == "negativa")
    neutras   = sum(1 for r in resultados if r["emocion_detectada"] == "neutra")

    riesgo_alto  = sum(1 for r in resultados if r["nivel_riesgo"] == "alto")
    riesgo_medio = sum(1 for r in resultados if r["nivel_riesgo"] == "medio")
    riesgo_bajo  = sum(1 for r in resultados if r["nivel_riesgo"] == "bajo")

    if negativas > positivas:
        conclusion    = "Predominan percepciones negativas o situaciones de malestar en la población analizada."
        recomendacion = "Se recomienda revisar los factores que generan insatisfacción y fortalecer estrategias de mejora."
    elif positivas > negativas:
        conclusion    = "Predominan percepciones positivas y satisfacción general en la población analizada."
        recomendacion = "Se recomienda mantener las buenas prácticas actuales y reforzar los factores positivos detectados."
    else:
        conclusion    = "Se observan percepciones mixtas o mayormente neutras en la población analizada."
        recomendacion = "Se recomienda profundizar el análisis con entrevistas focales para obtener mayor detalle."

    return {
        "emociones": {
            "positivas": positivas,
            "negativas": negativas,
            "neutras":   neutras
        },
        "niveles_riesgo": {
            "alto":  riesgo_alto,
            "medio": riesgo_medio,
            "bajo":  riesgo_bajo
        },
        "conclusion_general":    conclusion,
        "recomendacion_general": recomendacion
    }


def leer_csv_robusto(contenido: bytes) -> pd.DataFrame:
    """
    Intenta leer un CSV probando múltiples encodings.
    Cubre archivos exportados desde Excel en Windows (latin-1, cp1252)
    y archivos estándar (utf-8).
    """
    encodings = ["utf-8", "latin-1", "cp1252", "utf-8-sig", "iso-8859-1"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                io.BytesIO(contenido),
                encoding=encoding,
                on_bad_lines="skip"
            )
            if len(df.columns) > 0 and len(df) > 0:
                return df
        except Exception:
            continue

    raise HTTPException(
        status_code=400,
        detail="No se pudo leer el archivo CSV. Verifica que esté bien formado."
    )


def leer_archivo(archivo: UploadFile) -> List[str]:
    """
    Lee un archivo CSV, Excel o JSON.
    - CSV: prueba múltiples encodings automáticamente
    - Excel: lee TODAS las hojas y las combina
    - JSON: convierte lista de objetos a DataFrame
    - Detecta columnas con texto largo (respuestas abiertas)
    - Compatible con pandas 2.x y 3.x
    """
    contenido = archivo.file.read()
    nombre    = archivo.filename.lower()

    try:
        if nombre.endswith(".csv"):
            df = leer_csv_robusto(contenido)

        elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
            todas_hojas = pd.read_excel(io.BytesIO(contenido), sheet_name=None)
            df = pd.concat(todas_hojas.values(), ignore_index=True)

        elif nombre.endswith(".json"):
            datos = json.loads(contenido)
            if isinstance(datos, list):
                df = pd.DataFrame(datos)
            elif isinstance(datos, dict):
                df = pd.DataFrame([datos])
            else:
                raise HTTPException(status_code=400, detail="Formato JSON no reconocido.")

        else:
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Usa CSV, Excel (.xlsx) o JSON."
            )

        # Detectar columnas con texto largo (respuestas abiertas)
        columnas_texto = []
        for col in df.columns:
            muestra = df[col].dropna().astype(str)
            if len(muestra) == 0:
                continue
            # Compatible con pandas 2.x y 3.x
            if not pd.api.types.is_string_dtype(df[col]):
                continue
            longitud_promedio = muestra.apply(len).mean()
            # Umbral: promedio mayor a 30 caracteres
            if longitud_promedio > 30:
                columnas_texto.append(col)

        if not columnas_texto:
            raise HTTPException(
                status_code=400,
                detail="No se detectaron columnas con respuestas abiertas en el archivo."
            )

        respuestas = []
        for col in columnas_texto:
            valores = df[col].dropna().astype(str).tolist()
            respuestas.extend([v for v in valores if len(v.strip()) > 10])

        return respuestas

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer el archivo: {str(e)}"
        )


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/")
def inicio():
    return {
        "nombre":  "API de Análisis Cualitativo con IA",
        "version": "2.1.0",
        "estado":  "funcionando",
        "endpoints": {
            "analizar_individual":    "/analizar",
            "analizar_lote":          "/analizar-lote",
            "analizar_archivo":       "/analizar-archivo",
            "analizar_google_sheets": "/analizar-google-sheets",
            "documentacion":          "/docs"
        }
    }


@app.post("/analizar")
def analizar_respuesta(data: RespuestaInput):
    """Analiza una sola respuesta abierta."""

    if not data.texto or len(data.texto.strip()) < 3:
        raise HTTPException(status_code=400, detail="El texto está vacío o es demasiado corto.")

    palabras_clave = extraer_palabras_clave(data.texto)
    analisis       = analizar_con_groq(data.texto)

    return {
        **analisis,
        "palabras_clave":  palabras_clave,
        "texto_analizado": data.texto[:100] + "..." if len(data.texto) > 100 else data.texto
    }


@app.post("/analizar-lote")
def analizar_lote(data: LoteInput):
    """Analiza un lote de respuestas enviadas como JSON."""

    if not data.respuestas:
        raise HTTPException(status_code=400, detail="La lista de respuestas está vacía.")

    if len(data.respuestas) > 500:
        raise HTTPException(
            status_code=400,
            detail="Máximo 500 respuestas por lote. Para más usa /analizar-archivo."
        )

    todas_las_palabras      = []
    resultados_individuales = []

    for respuesta in data.respuestas:
        if not respuesta or len(respuesta.strip()) < 3:
            continue

        claves = extraer_palabras_clave(respuesta)
        todas_las_palabras.extend(claves)

        analisis = analizar_con_groq(respuesta, data.contexto)
        resultados_individuales.append(analisis)

    conteo_general    = Counter(todas_las_palabras)
    temas_principales = [palabra for palabra, _ in conteo_general.most_common(10)]
    resumen           = generar_resumen_lote(resultados_individuales, data.contexto)

    return {
        "total_respuestas":             len(resultados_individuales),
        "temas_principales_detectados": temas_principales,
        **resumen
    }


@app.post("/analizar-archivo")
async def analizar_archivo(
    archivo: UploadFile = File(...),
    contexto: str = "análisis cualitativo"
):
    """
    Acepta archivos CSV, Excel (.xlsx) o JSON.
    - CSV: detecta encoding automáticamente (utf-8, latin-1, cp1252, etc.)
    - Excel: lee todas las hojas automáticamente
    - Detecta columnas con respuestas abiertas sin configuración manual
    - Funciona para cualquier tipo de encuesta o tema
    """

    respuestas = leer_archivo(archivo)

    if not respuestas:
        raise HTTPException(
            status_code=400,
            detail="No se encontraron respuestas válidas en el archivo."
        )

    todas_las_palabras      = []
    resultados_individuales = []

    TAMANIO_LOTE = 50

    for i in range(0, len(respuestas), TAMANIO_LOTE):
        lote = respuestas[i:i + TAMANIO_LOTE]

        for respuesta in lote:
            claves = extraer_palabras_clave(respuesta)
            todas_las_palabras.extend(claves)

            analisis = analizar_con_groq(respuesta, contexto)
            resultados_individuales.append(analisis)

    conteo_general    = Counter(todas_las_palabras)
    temas_principales = [palabra for palabra, _ in conteo_general.most_common(10)]
    resumen           = generar_resumen_lote(resultados_individuales, contexto)

    return {
        "archivo":                      archivo.filename,
        "total_respuestas_procesadas":  len(resultados_individuales),
        "temas_principales_detectados": temas_principales,
        **resumen
    }


@app.post("/analizar-google-sheets")
def analizar_google_sheets(data: GoogleSheetsInput):
    """
    Recibe datos exportados desde Google Sheets o Google Forms.
    Espera una lista de diccionarios con una columna específica de respuestas.

    Ejemplo de body:
    {
        "datos": [{"respuesta": "Me siento muy estresado"}, ...],
        "columna_respuestas": "respuesta",
        "contexto": "encuesta de satisfacción de clientes"
    }
    """

    if not data.datos:
        raise HTTPException(status_code=400, detail="No se recibieron datos.")

    if data.columna_respuestas not in data.datos[0]:
        raise HTTPException(
            status_code=400,
            detail=f"La columna '{data.columna_respuestas}' no existe en los datos."
        )

    respuestas = [
        str(fila[data.columna_respuestas])
        for fila in data.datos
        if fila.get(data.columna_respuestas)
        and len(str(fila[data.columna_respuestas]).strip()) > 3
    ]

    if not respuestas:
        raise HTTPException(status_code=400, detail="No se encontraron respuestas válidas.")

    todas_las_palabras      = []
    resultados_individuales = []

    for respuesta in respuestas:
        claves = extraer_palabras_clave(respuesta)
        todas_las_palabras.extend(claves)

        analisis = analizar_con_groq(respuesta, data.contexto)
        resultados_individuales.append(analisis)

    conteo_general    = Counter(todas_las_palabras)
    temas_principales = [palabra for palabra, _ in conteo_general.most_common(10)]
    resumen           = generar_resumen_lote(resultados_individuales, data.contexto)

    return {
        "total_respuestas_procesadas":  len(resultados_individuales),
        "temas_principales_detectados": temas_principales,
        **resumen
    }
