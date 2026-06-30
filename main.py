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
    description="Plataforma de análisis cualitativo con filtrado inteligente de respuestas",
    version="4.1.0"
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

# --------------------------------------------------
# MODELOS DE ENTRADA
# --------------------------------------------------

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

class FiltrosAnalisis(BaseModel):
    detectar_irrelevantes: bool = True
    detectar_ofensivas: bool = True
    detectar_alertas: bool = True
    analisis_conductual: bool = True


# --------------------------------------------------
# CLASIFICACIÓN DE RESPUESTAS
# --------------------------------------------------

def clasificar_respuesta(texto: str, contexto: str) -> dict:
    """
    Clasifica una respuesta en:
    - valida: entra al análisis normal
    - irrelevante: sin relación con el contexto
    - ofensiva: insultos hacia institución o personas
    - alerta: señales de bienestar preocupantes
    Devuelve categoria + razon
    """
    prompt = f"""Eres un sistema experto en análisis de encuestas institucionales.

Contexto de la encuesta: {contexto}

Clasifica la siguiente respuesta en UNA de estas categorías:
- "valida": respuesta coherente y relacionada con el contexto
- "irrelevante": respuesta sin sentido, fuera de contexto, aleatoria o vacía de significado (ej: palabras sueltas sin relación, caracteres aleatorios, nombres de productos, etc.)
- "ofensiva": contiene insultos, lenguaje agresivo o despectivo hacia la institución o personas
- "alerta": contiene señales preocupantes de bienestar personal (autolesión, violencia, crisis emocional severa) independientemente del tema de la encuesta
- "conductual": respuesta que revela un patrón de comportamiento relevante (evasión, agresividad pasiva, apatía extrema, desmotivación profunda)

Responde ÚNICAMENTE con este JSON:
{{
  "categoria": "valida" | "irrelevante" | "ofensiva" | "alerta" | "conductual",
  "razon": "explicación breve de por qué se clasifica así"
}}

Respuesta a clasificar:
"{texto}"

Responde solo con el JSON:"""

    try:
        respuesta = cliente_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        contenido = respuesta.choices[0].message.content.strip()
        contenido = contenido.replace("```json", "").replace("```", "").strip()
        resultado = json.loads(contenido)
        return {
            "categoria": resultado.get("categoria", "valida"),
            "razon":     resultado.get("razon", "")
        }
    except Exception:
        return {"categoria": "valida", "razon": "No se pudo clasificar"}


def analizar_con_groq(texto: str, contexto: str = "análisis cualitativo") -> dict:
    prompt = f"""Eres un sistema experto en análisis cualitativo de texto.

Analiza la siguiente respuesta en el contexto de: {contexto}

Responde ÚNICAMENTE con este formato JSON, sin texto adicional:

{{
  "emocion": "positiva" | "negativa" | "neutra",
  "nivel_riesgo": "alto" | "medio" | "bajo",
  "confianza": número entre 0.0 y 1.0,
  "interpretacion": "breve explicación en español de máximo 2 oraciones",
  "patron_conductual": "ninguno" | "evasion" | "agresividad_pasiva" | "apatia" | "desmotivacion" | "cooperativo" | "critico_constructivo"
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
            max_tokens=250
        )
        contenido = respuesta.choices[0].message.content.strip()
        contenido = contenido.replace("```json", "").replace("```", "").strip()
        resultado = json.loads(contenido)
        return {
            "emocion_detectada":  resultado.get("emocion", "neutra"),
            "nivel_riesgo":       resultado.get("nivel_riesgo", "bajo"),
            "confianza_modelo":   round(float(resultado.get("confianza", 0.5)), 2),
            "interpretacion":     resultado.get("interpretacion", "Sin interpretación disponible."),
            "patron_conductual":  resultado.get("patron_conductual", "ninguno")
        }
    except Exception as e:
        return {
            "emocion_detectada":  "neutra",
            "nivel_riesgo":       "bajo",
            "confianza_modelo":   0.0,
            "interpretacion":     f"Error al analizar: {str(e)}",
            "patron_conductual":  "ninguno"
        }


def interpretar_individual_con_groq(texto: str, analisis: dict, palabras_clave: List[str], contexto: str) -> str:
    prompt = f"""Eres un experto en análisis cualitativo.

Contexto: {contexto}
Respuesta: "{texto}"
Emoción: {analisis['emocion_detectada']} | Riesgo: {analisis['nivel_riesgo']}
Patrón conductual: {analisis.get('patron_conductual', 'ninguno')}
Palabras clave: {', '.join(palabras_clave) if palabras_clave else 'ninguna'}

Genera una interpretación detallada de 3 a 4 oraciones. Explica el porqué de la emoción,
qué factores la generan, el patrón conductual si aplica, y qué acción concreta se recomienda.
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


def perfil_persona(positivas: int, negativas: int, neutras: int) -> str:
    total = positivas + negativas + neutras
    if total == 0:
        return "sin datos"
    pct_pos = positivas / total
    pct_neg = negativas / total
    if pct_pos >= 0.6:   return "mayormente positivo"
    elif pct_neg >= 0.6: return "mayormente negativo"
    elif pct_pos > pct_neg: return "tendencia positiva"
    elif pct_neg > pct_pos: return "tendencia negativa"
    else: return "mixto"


def generar_resumen_lote(resultados: List[dict], contexto: str, temas: List[str] = [],
                          patrones: dict = {}) -> dict:
    positivas    = sum(1 for r in resultados if r["emocion_detectada"] == "positiva")
    negativas    = sum(1 for r in resultados if r["emocion_detectada"] == "negativa")
    neutras      = sum(1 for r in resultados if r["emocion_detectada"] == "neutra")
    riesgo_alto  = sum(1 for r in resultados if r["nivel_riesgo"] == "alto")
    riesgo_medio = sum(1 for r in resultados if r["nivel_riesgo"] == "medio")
    riesgo_bajo  = sum(1 for r in resultados if r["nivel_riesgo"] == "bajo")
    total        = len(resultados)

    patrones_str = ", ".join([f"{k}: {v}" for k, v in patrones.items() if v > 0]) or "ninguno detectado"

    prompt = f"""Eres un experto en análisis cualitativo.
Se analizaron {total} respuestas válidas en el contexto de: {contexto}

Resultados:
- Emociones positivas: {positivas} | negativas: {negativas} | neutras: {neutras}
- Riesgo alto: {riesgo_alto} | medio: {riesgo_medio} | bajo: {riesgo_bajo}
- Temas principales: {', '.join(temas) if temas else 'no disponibles'}
- Patrones conductuales: {patrones_str}

Responde ÚNICAMENTE con este JSON:
{{
  "conclusion_general": "conclusión específica de 2-3 oraciones considerando emociones y patrones conductuales",
  "recomendacion_general": "recomendación concreta de 2-3 oraciones adaptada al contexto"
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
        conclusion    = "Se completó el análisis de las respuestas válidas."
        recomendacion = "Revisa los resultados detallados para tomar decisiones informadas."

    return {
        "emociones":      {"positivas": positivas, "negativas": negativas, "neutras": neutras},
        "niveles_riesgo": {"alto": riesgo_alto, "medio": riesgo_medio, "bajo": riesgo_bajo},
        "patrones_conductuales": patrones,
        "conclusion_general":    conclusion,
        "recomendacion_general": recomendacion
    }


def detectar_columnas_identidad(df: pd.DataFrame) -> dict:
    palabras_nombre = ["nombre", "name", "alumno", "empleado", "participante",
                       "encuestado", "sujeto", "trabajador", "estudiante", "usuario", "apellido"]
    palabras_id     = ["id", "matricula", "matrícula", "clave", "codigo", "código",
                       "folio", "numero", "número", "num", "expediente", "cuenta"]
    col_nombre = None
    col_id     = None

    for col in df.columns:
        nombre_col = str(col).lower()
        if col_nombre is None and any(k in nombre_col for k in palabras_nombre):
            col_nombre = col
        if col_id is None and any(k in nombre_col for k in palabras_id):
            col_id = col

    if col_nombre is None:
        for col in df.columns:
            if col == col_id: continue
            muestra = df[col].dropna().astype(str)
            if len(muestra) == 0: continue
            es_numerica = pd.to_numeric(muestra, errors='coerce').notna().mean() > 0.8
            if muestra.apply(len).mean() < 35 and muestra.nunique()/len(muestra) > 0.7 and not es_numerica:
                col_nombre = col; break

    if col_id is None:
        for col in df.columns:
            if col == col_nombre: continue
            muestra = df[col].dropna().astype(str)
            if len(muestra) == 0: continue
            es_numerica = pd.to_numeric(muestra, errors='coerce').notna().mean() > 0.8
            if es_numerica and muestra.nunique()/len(muestra) > 0.7:
                col_id = col; break

    return {"nombre": col_nombre, "id": col_id}


def construir_identificador(fila, cols_identidad: dict) -> Optional[str]:
    col_nombre = cols_identidad.get("nombre")
    col_id     = cols_identidad.get("id")
    nombre = str(fila[col_nombre]).strip() if col_nombre and pd.notna(fila.get(col_nombre)) else None
    id_val = str(fila[col_id]).strip()     if col_id     and pd.notna(fila.get(col_id))     else None
    if nombre and nombre.lower() in ["nan","","none"]: nombre = None
    if id_val and id_val.lower() in ["nan","","none"]: id_val = None
    if id_val:
        try: id_val = str(int(float(id_val)))
        except: pass
    if nombre and id_val:
        return f"{nombre} ({str(col_id).strip()}: {id_val})"
    elif nombre: return nombre
    elif id_val: return f"{str(col_id).strip() if col_id else 'ID'}: {id_val}"
    else: return None


def leer_csv_robusto(contenido: bytes) -> pd.DataFrame:
    for encoding in ["utf-8","latin-1","cp1252","utf-8-sig","iso-8859-1"]:
        try:
            df = pd.read_csv(io.BytesIO(contenido), encoding=encoding, on_bad_lines="skip")
            if len(df.columns) > 0 and len(df) > 0: return df
        except: continue
    raise HTTPException(status_code=400, detail="No se pudo leer el archivo CSV.")


def leer_dataframes(archivo: UploadFile) -> List[tuple]:
    """Devuelve lista de tuplas (nombre_hoja, DataFrame)."""
    contenido = archivo.file.read()
    nombre    = archivo.filename.lower()
    if nombre.endswith(".csv"):
        return [("Datos", leer_csv_robusto(contenido))]
    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        todas_hojas = pd.read_excel(io.BytesIO(contenido), sheet_name=None)
        return list(todas_hojas.items())  # (nombre_hoja, df)
    elif nombre.endswith(".json"):
        datos = json.loads(contenido)
        return [("Datos", pd.DataFrame(datos if isinstance(datos, list) else [datos]))]
    else:
        raise HTTPException(status_code=400, detail="Formato no soportado.")


def detectar_columnas_respuestas(df: pd.DataFrame, excluir: List[str] = []) -> List[str]:
    columnas = []
    for col in df.columns:
        if col in excluir: continue
        muestra = df[col].dropna().astype(str)
        if len(muestra) == 0 or not pd.api.types.is_string_dtype(df[col]): continue
        if muestra.apply(len).mean() > 30: columnas.append(col)
    return columnas


def procesar_dataframe(df: pd.DataFrame, contexto: str, filtros: dict = {}) -> Optional[dict]:
    cols_identidad      = detectar_columnas_identidad(df)
    col_nombre          = cols_identidad.get("nombre")
    col_id              = cols_identidad.get("id")
    excluir             = [c for c in [col_nombre, col_id] if c is not None]
    columnas_respuestas = detectar_columnas_respuestas(df, excluir=excluir)

    if not columnas_respuestas:
        return None

    # Filtros activos
    detectar_irrelevantes = filtros.get("detectar_irrelevantes", True)
    detectar_ofensivas    = filtros.get("detectar_ofensivas", True)
    detectar_alertas      = filtros.get("detectar_alertas", True)
    analisis_conductual   = filtros.get("analisis_conductual", True)

    todas_las_palabras  = []
    resultados_validos  = []
    irrelevantes        = []
    ofensivas           = []
    alertas             = []
    conductuales        = []
    patrones_contador   = Counter()

    hay_identidad = col_nombre is not None or col_id is not None

    if hay_identidad:
        personas_resultado = []

        for _, fila in df.iterrows():
            identificador = construir_identificador(fila, cols_identidad)
            if identificador is None:
                continue

            pos = neg = neu = 0
            riesgo_max       = "bajo"
            interpretaciones = []

            for col in columnas_respuestas:
                texto = str(fila[col]) if pd.notna(fila.get(col)) else ""
                if len(texto.strip()) < 3:
                    continue

                # Clasificar respuesta primero
                clasificacion = clasificar_respuesta(texto, contexto)
                categoria     = clasificacion["categoria"]
                razon         = clasificacion["razon"]

                entrada_filtrada = {
                    "persona": identificador,
                    "texto":   texto,
                    "razon":   razon
                }

                if categoria == "irrelevante" and detectar_irrelevantes:
                    irrelevantes.append(entrada_filtrada)
                    continue
                elif categoria == "ofensiva" and detectar_ofensivas:
                    ofensivas.append(entrada_filtrada)
                    continue
                elif categoria == "alerta" and detectar_alertas:
                    alertas.append(entrada_filtrada)
                    continue

                # Analizar respuestas válidas y conductuales
                todas_las_palabras.extend(extraer_palabras_clave(texto))
                analisis = analizar_con_groq(texto, contexto)
                resultados_validos.append(analisis)

                if analisis_conductual and analisis.get("patron_conductual", "ninguno") != "ninguno":
                    patrones_contador[analisis["patron_conductual"]] += 1
                    if categoria == "conductual":
                        conductuales.append({**entrada_filtrada, "patron": analisis["patron_conductual"]})

                if analisis["emocion_detectada"] == "positiva": pos += 1
                elif analisis["emocion_detectada"] == "negativa": neg += 1
                else: neu += 1

                if analisis["nivel_riesgo"] == "alto": riesgo_max = "alto"
                elif analisis["nivel_riesgo"] == "medio" and riesgo_max != "alto": riesgo_max = "medio"
                interpretaciones.append(analisis["interpretacion"])

            if pos + neg + neu == 0:
                continue

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
        resumen = generar_resumen_lote(resultados_validos, contexto, temas, dict(patrones_contador))

        grupo_positivo = [p["persona"] for p in personas_resultado if "positiv" in p["perfil_emocional"]]
        grupo_negativo = [p["persona"] for p in personas_resultado if "negativ" in p["perfil_emocional"]]
        grupo_mixto    = [p["persona"] for p in personas_resultado if p["perfil_emocional"] in ["mixto","sin datos"]]

        return {
            "columna_nombre_detectada":     str(col_nombre) if col_nombre else None,
            "columna_id_detectada":         str(col_id)     if col_id     else None,
            "total_personas":               len(personas_resultado),
            "total_respuestas_validas":     len(resultados_validos),
            "total_respuestas_filtradas":   len(irrelevantes) + len(ofensivas) + len(alertas),
            "temas_principales_detectados": temas,
            "analisis_por_persona":         personas_resultado,
            "grupos": {
                "perfil_positivo": grupo_positivo,
                "perfil_negativo": grupo_negativo,
                "perfil_mixto":    grupo_mixto
            },
            "respuestas_filtradas": {
                "irrelevantes": irrelevantes,
                "ofensivas":    ofensivas,
                "alertas":      alertas,
                "conductuales": conductuales
            },
            **resumen
        }

    else:
        # Sin identidad — análisis general con filtrado
        for col in columnas_respuestas:
            for texto in df[col].dropna().astype(str).tolist():
                if len(texto.strip()) < 3:
                    continue

                clasificacion = clasificar_respuesta(texto, contexto)
                categoria     = clasificacion["categoria"]
                razon         = clasificacion["razon"]
                entrada        = {"texto": texto, "razon": razon}

                if categoria == "irrelevante" and detectar_irrelevantes:
                    irrelevantes.append(entrada); continue
                elif categoria == "ofensiva" and detectar_ofensivas:
                    ofensivas.append(entrada); continue
                elif categoria == "alerta" and detectar_alertas:
                    alertas.append(entrada); continue

                todas_las_palabras.extend(extraer_palabras_clave(texto))
                analisis = analizar_con_groq(texto, contexto)
                resultados_validos.append(analisis)

                if analisis_conductual and analisis.get("patron_conductual","ninguno") != "ninguno":
                    patrones_contador[analisis["patron_conductual"]] += 1
                    if categoria == "conductual":
                        conductuales.append({**entrada, "patron": analisis["patron_conductual"]})

        temas   = [p for p, _ in Counter(todas_las_palabras).most_common(10)]
        resumen = generar_resumen_lote(resultados_validos, contexto, temas, dict(patrones_contador))

        return {
            "columna_nombre_detectada":     None,
            "columna_id_detectada":         None,
            "total_respuestas_validas":     len(resultados_validos),
            "total_respuestas_filtradas":   len(irrelevantes) + len(ofensivas) + len(alertas),
            "temas_principales_detectados": temas,
            "respuestas_filtradas": {
                "irrelevantes": irrelevantes,
                "ofensivas":    ofensivas,
                "alertas":      alertas,
                "conductuales": conductuales
            },
            **resumen
        }


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/")
def inicio():
    return {
        "nombre":  "API de Análisis Cualitativo con IA",
        "version": "4.1.0",
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
    """Analiza una sola respuesta con clasificación, filtrado e interpretación enriquecida."""
    if not data.texto or len(data.texto.strip()) < 3:
        raise HTTPException(status_code=400, detail="El texto está vacío o es demasiado corto.")
    contexto       = data.contexto or "análisis cualitativo"
    clasificacion  = clasificar_respuesta(data.texto, contexto)
    palabras_clave = extraer_palabras_clave(data.texto)
    analisis       = analizar_con_groq(data.texto, contexto)
    interpretacion = interpretar_individual_con_groq(data.texto, analisis, palabras_clave, contexto)
    return {
        "categoria":         clasificacion["categoria"],
        "razon_categoria":   clasificacion["razon"],
        "emocion_detectada": analisis["emocion_detectada"],
        "nivel_riesgo":      analisis["nivel_riesgo"],
        "confianza_modelo":  analisis["confianza_modelo"],
        "patron_conductual": analisis.get("patron_conductual", "ninguno"),
        "interpretacion":    interpretacion,
        "palabras_clave":    palabras_clave,
        "texto_analizado":   data.texto[:100] + "..." if len(data.texto) > 100 else data.texto
    }


@app.post("/analizar-lote")
def analizar_lote(data: LoteInput):
    """Analiza un lote de respuestas con filtrado inteligente."""
    if not data.respuestas:
        raise HTTPException(status_code=400, detail="La lista de respuestas está vacía.")
    if len(data.respuestas) > 500:
        raise HTTPException(status_code=400, detail="Máximo 500 respuestas.")

    todas_las_palabras = []
    resultados         = []
    irrelevantes       = []
    ofensivas          = []
    alertas            = []
    patrones_contador  = Counter()

    for respuesta in data.respuestas:
        if not respuesta or len(respuesta.strip()) < 3: continue
        clasificacion = clasificar_respuesta(respuesta, data.contexto)
        categoria     = clasificacion["categoria"]
        entrada       = {"texto": respuesta, "razon": clasificacion["razon"]}

        if categoria == "irrelevante": irrelevantes.append(entrada); continue
        elif categoria == "ofensiva":  ofensivas.append(entrada);    continue
        elif categoria == "alerta":    alertas.append(entrada);      continue

        todas_las_palabras.extend(extraer_palabras_clave(respuesta))
        analisis = analizar_con_groq(respuesta, data.contexto)
        resultados.append(analisis)
        if analisis.get("patron_conductual","ninguno") != "ninguno":
            patrones_contador[analisis["patron_conductual"]] += 1

    temas   = [p for p, _ in Counter(todas_las_palabras).most_common(10)]
    resumen = generar_resumen_lote(resultados, data.contexto, temas, dict(patrones_contador))

    return {
        "total_respuestas_validas":   len(resultados),
        "total_respuestas_filtradas": len(irrelevantes)+len(ofensivas)+len(alertas),
        "temas_principales_detectados": temas,
        "respuestas_filtradas": {
            "irrelevantes": irrelevantes,
            "ofensivas":    ofensivas,
            "alertas":      alertas
        },
        **resumen
    }


@app.post("/analizar-archivo")
async def analizar_archivo(
    archivo: UploadFile = File(...),
    contexto: str = "análisis cualitativo",
    detectar_irrelevantes: bool = True,
    detectar_ofensivas: bool = True,
    detectar_alertas: bool = True,
    analisis_conductual: bool = True
):
    """
    Acepta CSV, Excel o JSON.
    Filtra automáticamente respuestas irrelevantes, ofensivas, alertas y conductuales.
    Procesa cada hoja de Excel por separado.
    """
    filtros = {
        "detectar_irrelevantes": detectar_irrelevantes,
        "detectar_ofensivas":    detectar_ofensivas,
        "detectar_alertas":      detectar_alertas,
        "analisis_conductual":   analisis_conductual
    }

    try:
        dataframes = leer_dataframes(archivo)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo: {str(e)}")

    if not dataframes:
        raise HTTPException(status_code=400, detail="No se encontraron datos en el archivo.")

    # Si solo hay una hoja/fuente, devolver resultado directo (compatibilidad)
    if len(dataframes) == 1:
        nombre_hoja, df = dataframes[0]
        resultado = procesar_dataframe(df, contexto, filtros)
        if resultado is None:
            raise HTTPException(status_code=400, detail="No se detectaron columnas con respuestas abiertas.")
        resultado["archivo"]     = archivo.filename
        resultado["nombre_hoja"] = nombre_hoja
        return resultado

    # Múltiples hojas — procesar cada una con su nombre real
    resultados_por_hoja = []
    for nombre_hoja, df in dataframes:
        resultado = procesar_dataframe(df, contexto, filtros)
        if resultado is None:
            continue
        resultado["nombre_hoja"] = nombre_hoja
        resultados_por_hoja.append(resultado)

    if not resultados_por_hoja:
        raise HTTPException(status_code=400, detail="No se detectaron respuestas abiertas en ninguna hoja.")

    return {
        "archivo":             archivo.filename,
        "total_hojas":         len(resultados_por_hoja),
        "resultados_por_hoja": resultados_por_hoja
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
    irrelevantes       = []
    ofensivas          = []
    alertas            = []
    patrones_contador  = Counter()

    for respuesta in respuestas:
        clasificacion = clasificar_respuesta(respuesta, data.contexto)
        categoria     = clasificacion["categoria"]
        entrada       = {"texto": respuesta, "razon": clasificacion["razon"]}

        if categoria == "irrelevante": irrelevantes.append(entrada); continue
        elif categoria == "ofensiva":  ofensivas.append(entrada);    continue
        elif categoria == "alerta":    alertas.append(entrada);      continue

        todas_las_palabras.extend(extraer_palabras_clave(respuesta))
        analisis = analizar_con_groq(respuesta, data.contexto)
        resultados.append(analisis)
        if analisis.get("patron_conductual","ninguno") != "ninguno":
            patrones_contador[analisis["patron_conductual"]] += 1

    temas   = [p for p, _ in Counter(todas_las_palabras).most_common(10)]
    resumen = generar_resumen_lote(resultados, data.contexto, temas, dict(patrones_contador))

    return {
        "total_respuestas_validas":     len(resultados),
        "total_respuestas_filtradas":   len(irrelevantes)+len(ofensivas)+len(alertas),
        "temas_principales_detectados": temas,
        "respuestas_filtradas": {"irrelevantes": irrelevantes, "ofensivas": ofensivas, "alertas": alertas},
        **resumen
    }
