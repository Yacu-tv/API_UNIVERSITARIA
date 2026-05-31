from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from collections import Counter
from dotenv import load_dotenv
import spacy
import pandas as pd
import groq
import os
import io
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("No se encontró GROQ_API_KEY en el archivo .env")

cliente_groq = groq.Groq(api_key=GROQ_API_KEY)
nlp = spacy.load("es_core_news_sm")

app = FastAPI(
    title="API de Análisis Cualitativo con IA",
    description="Plataforma de análisis cualitativo para cualquier tipo de encuesta o tema",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STOPWORDS = nlp.Defaults.stop_words | {
    "siento", "estoy", "existe", "mucha", "mucho", "tantas",
    "ultimamente", "tener", "tengo", "hacer", "puede", "pueden",
    "persona", "personas", "cosas", "tema", "temas", "bien", "mal"
}

class RespuestaInput(BaseModel):
    texto: str
    contexto: Optional[str] = "análisis cualitativo"

class LoteInput(BaseModel):
    respuestas: List[str]
    contexto: Optional[str] = "análisis cualitativo"

class GoogleSheetsInput(BaseModel):
    datos: List[dict]
    columna_respuestas: str
    contexto: Optional[str] = "análisis cualitativo"


def extraer_palabras_clave(texto: str) -> List[str]:
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
        return [p for p, _ in Counter(palabras_clave).most_common(5)]
    except Exception:
        return []


def analizar_con_groq(texto: str, contexto: str = "análisis cualitativo") -> dict:
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
            "nivel_riesgo":      resultado.get("nivel_riesgo", "bajo"),
            "confianza_modelo":  round(float(resultado.get("confianza", 0.5)), 2),
            "interpretacion":    resultado.get("interpretacion", "Sin interpretación disponible.")
        }
    except Exception as e:
        return {
            "emocion_detectada": "neutra",
            "nivel_riesgo":      "bajo",
            "confianza_modelo":  0.0,
            "interpretacion":    f"Error al analizar: {str(e)}"
        }


def interpretar_individual_con_groq(texto: str, analisis: dict, palabras_clave: List[str], contexto: str) -> str:
    prompt = f"""Eres un experto en análisis cualitativo.

Contexto: {contexto}
Respuesta: "{texto}"
Emoción: {analisis['emocion_detectada']} | Riesgo: {analisis['nivel_riesgo']}
Palabras clave: {', '.join(palabras_clave) if palabras_clave else 'ninguna'}

Genera una interpretación detallada de 3 a 4 oraciones. Explica el porqué de la emoción,
qué factores la generan y qué acción concreta se recomienda.
Responde SOLO con el texto, sin JSON ni formato especial:"""

    try:
        respuesta = cliente_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250
        )
        return respuesta.choices[0].message.content.strip()
    except Exception:
        return analisis["interpretacion"]


def perfil_persona(positivas: int, negativas: int, neutras: int) -> str:
    total = positivas + negativas + neutras
    if total == 0:
        return "sin datos"
    pct_pos = positivas / total
    pct_neg = negativas / total
    if pct_pos >= 0.6:
        return "mayormente positivo"
    elif pct_neg >= 0.6:
        return "mayormente negativo"
    elif pct_pos > pct_neg:
        return "tendencia positiva"
    elif pct_neg > pct_pos:
        return "tendencia negativa"
    else:
        return "mixto"


def generar_resumen_lote(resultados: List[dict], contexto: str, temas: List[str] = []) -> dict:
    positivas    = sum(1 for r in resultados if r["emocion_detectada"] == "positiva")
    negativas    = sum(1 for r in resultados if r["emocion_detectada"] == "negativa")
    neutras      = sum(1 for r in resultados if r["emocion_detectada"] == "neutra")
    riesgo_alto  = sum(1 for r in resultados if r["nivel_riesgo"] == "alto")
    riesgo_medio = sum(1 for r in resultados if r["nivel_riesgo"] == "medio")
    riesgo_bajo  = sum(1 for r in resultados if r["nivel_riesgo"] == "bajo")
    total        = len(resultados)

    prompt = f"""Eres un experto en análisis cualitativo.
Se analizaron {total} respuestas en el contexto de: {contexto}

Resultados:
- Emociones positivas: {positivas} | negativas: {negativas} | neutras: {neutras}
- Riesgo alto: {riesgo_alto} | medio: {riesgo_medio} | bajo: {riesgo_bajo}
- Temas principales: {', '.join(temas) if temas else 'no disponibles'}

Responde ÚNICAMENTE con este JSON:
{{
  "conclusion_general": "conclusión específica de 2-3 oraciones",
  "recomendacion_general": "recomendación concreta de 2-3 oraciones"
}}"""

    try:
        respuesta = cliente_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        contenido = respuesta.choices[0].message.content.strip()
        contenido = contenido.replace("```json", "").replace("```", "").strip()
        resultado = json.loads(contenido)
        conclusion    = resultado.get("conclusion_general", "Análisis completado.")
        recomendacion = resultado.get("recomendacion_general", "Revisar resultados detallados.")
    except Exception:
        conclusion    = "Se completó el análisis de las respuestas proporcionadas."
        recomendacion = "Revisa los resultados detallados para tomar decisiones informadas."

    return {
        "emociones": {"positivas": positivas, "negativas": negativas, "neutras": neutras},
        "niveles_riesgo": {"alto": riesgo_alto, "medio": riesgo_medio, "bajo": riesgo_bajo},
        "conclusion_general":    conclusion,
        "recomendacion_general": recomendacion
    }


def detectar_columna_identidad(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        muestra = df[col].dropna().astype(str)
        if len(muestra) == 0:
            continue
        longitud_promedio = muestra.apply(len).mean()
        unicidad = muestra.nunique() / len(muestra)
        nombre_col = str(col).lower()
        if longitud_promedio < 30 and unicidad > 0.7:
            if any(k in nombre_col for k in ["nombre", "name", "id", "matricula", "matrícula", "alumno", "empleado", "participante"]):
                return col
    for col in df.columns:
        muestra = df[col].dropna().astype(str)
        if len(muestra) == 0:
            continue
        if muestra.apply(len).mean() < 30 and muestra.nunique() / len(muestra) > 0.7:
            return col
    return None


def leer_csv_robusto(contenido: bytes) -> pd.DataFrame:
    for encoding in ["utf-8", "latin-1", "cp1252", "utf-8-sig", "iso-8859-1"]:
        try:
            df = pd.read_csv(io.BytesIO(contenido), encoding=encoding, on_bad_lines="skip")
            if len(df.columns) > 0 and len(df) > 0:
                return df
        except Exception:
            continue
    raise HTTPException(status_code=400, detail="No se pudo leer el archivo CSV.")


def leer_dataframe(archivo: UploadFile) -> pd.DataFrame:
    contenido = archivo.file.read()
    nombre    = archivo.filename.lower()
    if nombre.endswith(".csv"):
        return leer_csv_robusto(contenido)
    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        todas_hojas = pd.read_excel(io.BytesIO(contenido), sheet_name=None)
        return pd.concat(todas_hojas.values(), ignore_index=True)
    elif nombre.endswith(".json"):
        datos = json.loads(contenido)
        return pd.DataFrame(datos if isinstance(datos, list) else [datos])
    else:
        raise HTTPException(status_code=400, detail="Formato no soportado. Usa CSV, Excel o JSON.")


def detectar_columnas_respuestas(df: pd.DataFrame) -> List[str]:
    columnas = []
    for col in df.columns:
        muestra = df[col].dropna().astype(str)
        if len(muestra) == 0 or not pd.api.types.is_string_dtype(df[col]):
            continue
        if muestra.apply(len).mean() > 30:
            columnas.append(col)
    return columnas


@app.get("/")
def inicio():
    return {
        "nombre":  "API de Análisis Cualitativo con IA",
        "version": "3.0.0",
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
    """Analiza una sola respuesta con interpretación enriquecida por IA."""
    if not data.texto or len(data.texto.strip()) < 3:
        raise HTTPException(status_code=400, detail="El texto está vacío o es demasiado corto.")
    contexto       = data.contexto or "análisis cualitativo"
    palabras_clave = extraer_palabras_clave(data.texto)
    analisis       = analizar_con_groq(data.texto, contexto)
    interpretacion = interpretar_individual_con_groq(data.texto, analisis, palabras_clave, contexto)
    return {
        "emocion_detectada": analisis["emocion_detectada"],
        "nivel_riesgo":      analisis["nivel_riesgo"],
        "confianza_modelo":  analisis["confianza_modelo"],
        "interpretacion":    interpretacion,
        "palabras_clave":    palabras_clave,
        "texto_analizado":   data.texto[:100] + "..." if len(data.texto) > 100 else data.texto
    }


@app.post("/analizar-lote")
def analizar_lote(data: LoteInput):
    """Analiza un lote de respuestas enviadas como JSON."""
    if not data.respuestas:
        raise HTTPException(status_code=400, detail="La lista de respuestas está vacía.")
    if len(data.respuestas) > 500:
        raise HTTPException(status_code=400, detail="Máximo 500 respuestas. Para más usa /analizar-archivo.")
    todas_las_palabras = []
    resultados         = []
    for respuesta in data.respuestas:
        if not respuesta or len(respuesta.strip()) < 3:
            continue
        todas_las_palabras.extend(extraer_palabras_clave(respuesta))
        resultados.append(analizar_con_groq(respuesta, data.contexto))
    temas   = [p for p, _ in Counter(todas_las_palabras).most_common(10)]
    resumen = generar_resumen_lote(resultados, data.contexto, temas)
    return {"total_respuestas": len(resultados), "temas_principales_detectados": temas, **resumen}


@app.post("/analizar-archivo")
async def analizar_archivo(
    archivo: UploadFile = File(...),
    contexto: str = "análisis cualitativo"
):
    """
    Acepta CSV, Excel o JSON.
    - Si detecta columna de identidad (nombre/ID/matrícula): análisis por persona + grupos
    - Si no: análisis general del grupo
    """
    try:
        df = leer_dataframe(archivo)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo: {str(e)}")

    columnas_respuestas = detectar_columnas_respuestas(df)
    if not columnas_respuestas:
        raise HTTPException(status_code=400, detail="No se detectaron columnas con respuestas abiertas.")

    col_identidad      = detectar_columna_identidad(df)
    todas_las_palabras = []
    resultados_globales = []

    if col_identidad:
        personas_resultado = []

        for _, fila in df.iterrows():
            identificador = str(fila[col_identidad]) if pd.notna(fila[col_identidad]) else None
            if not identificador or identificador.strip() == "":
                continue

            pos = neg = neu = 0
            riesgo_max = "bajo"
            interpretaciones = []

            for col in columnas_respuestas:
                texto = str(fila[col]) if pd.notna(fila[col]) else ""
                if len(texto.strip()) < 5:
                    continue
                todas_las_palabras.extend(extraer_palabras_clave(texto))
                analisis = analizar_con_groq(texto, contexto)
                resultados_globales.append(analisis)

                if analisis["emocion_detectada"] == "positiva":
                    pos += 1
                elif analisis["emocion_detectada"] == "negativa":
                    neg += 1
                else:
                    neu += 1

                if analisis["nivel_riesgo"] == "alto":
                    riesgo_max = "alto"
                elif analisis["nivel_riesgo"] == "medio" and riesgo_max != "alto":
                    riesgo_max = "medio"

                interpretaciones.append(analisis["interpretacion"])

            perfil = perfil_persona(pos, neg, neu)
            personas_resultado.append({
                "persona":              identificador,
                "respuestas_positivas": pos,
                "respuestas_negativas": neg,
                "respuestas_neutras":   neu,
                "perfil_emocional":     perfil,
                "nivel_riesgo_general": riesgo_max,
                "resumen_respuestas":   " | ".join(interpretaciones[:2])
            })

        temas   = [p for p, _ in Counter(todas_las_palabras).most_common(10)]
        resumen = generar_resumen_lote(resultados_globales, contexto, temas)

        grupo_positivo = [p["persona"] for p in personas_resultado if "positiv" in p["perfil_emocional"]]
        grupo_negativo = [p["persona"] for p in personas_resultado if "negativ" in p["perfil_emocional"]]
        grupo_mixto    = [p["persona"] for p in personas_resultado if p["perfil_emocional"] in ["mixto", "sin datos"]]

        return {
            "archivo":                      archivo.filename,
            "columna_identidad_detectada":  str(col_identidad),
            "total_personas":               len(personas_resultado),
            "total_respuestas_procesadas":  len(resultados_globales),
            "temas_principales_detectados": temas,
            "analisis_por_persona":         personas_resultado,
            "grupos": {
                "perfil_positivo": grupo_positivo,
                "perfil_negativo": grupo_negativo,
                "perfil_mixto":    grupo_mixto
            },
            **resumen
        }

    else:
        respuestas = []
        for col in columnas_respuestas:
            respuestas.extend([v for v in df[col].dropna().astype(str).tolist() if len(v.strip()) > 10])
        for respuesta in respuestas:
            todas_las_palabras.extend(extraer_palabras_clave(respuesta))
            resultados_globales.append(analizar_con_groq(respuesta, contexto))
        temas   = [p for p, _ in Counter(todas_las_palabras).most_common(10)]
        resumen = generar_resumen_lote(resultados_globales, contexto, temas)
        return {
            "archivo":                      archivo.filename,
            "columna_identidad_detectada":  None,
            "total_respuestas_procesadas":  len(resultados_globales),
            "temas_principales_detectados": temas,
            **resumen
        }


@app.post("/analizar-google-sheets")
def analizar_google_sheets(data: GoogleSheetsInput):
    """Recibe datos exportados desde Google Sheets o Google Forms."""
    if not data.datos:
        raise HTTPException(status_code=400, detail="No se recibieron datos.")
    if data.columna_respuestas not in data.datos[0]:
        raise HTTPException(status_code=400, detail=f"La columna '{data.columna_respuestas}' no existe.")
    respuestas = [
        str(fila[data.columna_respuestas])
        for fila in data.datos
        if fila.get(data.columna_respuestas) and len(str(fila[data.columna_respuestas]).strip()) > 3
    ]
    if not respuestas:
        raise HTTPException(status_code=400, detail="No se encontraron respuestas válidas.")
    todas_las_palabras = []
    resultados         = []
    for respuesta in respuestas:
        todas_las_palabras.extend(extraer_palabras_clave(respuesta))
        resultados.append(analizar_con_groq(respuesta, data.contexto))
    temas   = [p for p, _ in Counter(todas_las_palabras).most_common(10)]
    resumen = generar_resumen_lote(resultados, data.contexto, temas)
    return {"total_respuestas_procesadas": len(resultados), "temas_principales_detectados": temas, **resumen}
