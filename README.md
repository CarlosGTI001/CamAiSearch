# CamAiSearch# 🧠 AI Video Intelligence System

Sistema avanzado de análisis inteligente de video en tiempo real usando IA.  
Convierte cámaras de seguridad y videos en una plataforma capaz de **entender, describir y buscar eventos usando lenguaje natural**.

---

## 🚀 Descripción

Este proyecto permite analizar video en vivo o grabaciones para:

- Detectar personas y objetos
- Reconocer rostros conocidos
- Seguir individuos en tiempo real
- Generar descripciones automáticas de lo que ocurre
- Buscar eventos usando texto natural (tipo ChatGPT)
- Monitoreo en vivo con alertas inteligentes

---

## 🎯 Casos de uso

- Videovigilancia inteligente (CCTV con IA)
- Seguridad empresarial
- Análisis de comportamiento
- Investigación forense en video
- Control de accesos con reconocimiento facial
- Conteo y seguimiento de personas

---

## 🔍 Búsqueda inteligente (IA)

Puedes hacer preguntas como:

- "Persona con camiseta roja"
- "Alguien entrando por la puerta"
- "Persona dejando un objeto"
- "Personas corriendo"
- "Cuándo apareció Carlos"
- "Momentos sospechosos"

El sistema busca dentro del video usando IA multimodal y embeddings.

---

## 🧠 Características principales

### 🎥 Video Intelligence
- Detección de personas y objetos (YOLO)
- Tracking entre frames (ByteTrack / DeepSORT)
- Reconocimiento facial (InsightFace)
- Detección de eventos automáticos

### 🧾 Generación de eventos
- Timeline con timestamps reales
- Descripción automática de escenas
- Registro de actividad en JSON / logs / DB

### 🔎 Búsqueda semántica
- Embeddings con CLIP
- Búsqueda por texto natural
- Comparación entre video y lenguaje

### 📡 Monitoreo en tiempo real
- Cámaras RTSP / IP / USB
- Visualización en vivo
- Alertas instantáneas
- Múltiples cámaras simultáneas

### 🌐 API y sistema web
- FastAPI backend
- WebSockets en tiempo real
- Preparado para dashboard
- Sistema de alertas

---

## 🏗️ Arquitectura

```bash
project/
├── detector/ # YOLO detección de objetos
├── tracking/ # Seguimiento de personas
├── face/ # Reconocimiento facial
├── search/ # Búsqueda semántica
├── embeddings/ # CLIP / vectores IA
├── llm/ # Descripción de escenas
├── events/ # Generación de eventos
├── database/ # SQLite / PostgreSQL
├── api/ # FastAPI server
├── config/ # Configuración
├── output/ # Logs y resultados
├── videos/ # Videos de entrada
└── main.py
```

---

## 🛠️ Tecnologías

- Python 3.12+
- OpenCV
- Ultralytics YOLOv8 / YOLO11
- InsightFace
- ByteTrack / DeepSORT
- CLIP / Transformers
- FastAPI
- Whisper (opcional audio)
- PyTorch + CUDA
- SQLite / PostgreSQL

---

## ⚡ Ejemplo de eventos

```bash
00:00:03 - Persona entra al local
00:00:08 - Persona detectada: camiseta roja
00:00:15 - Objeto tomado del escritorio
00:00:22 - Persona sale del área
```

---

## 📡 API Endpoints (ejemplo)

- `POST /analyze` → analizar video
- `GET /events` → obtener eventos
- `GET /search?q=` → búsqueda inteligente
- `GET /live` → stream en vivo
- `GET /faces` → rostros detectados

---

## ⚙️ Instalación

```bash
git clone https://github.com/CarlosGTI001/CamAiSearch
cd CamAiSearch

pip install -r requirements.txt
```

Ejecutar:

```bash
python main.py
```

🔥 Objetivo

Transformar cualquier sistema de cámaras en una plataforma de inteligencia visual avanzada, capaz de:

Entender lo que ocurre en video
Responder preguntas en lenguaje natural
Detectar eventos automáticamente
Monitorear múltiples cámaras en tiempo real
🚧 Estado del proyecto

En desarrollo 🚀
Arquitectura base en construcción.

📌 Futuras mejoras
Dashboard web completo
Notificaciones (Telegram / Discord)
Heatmaps de movimiento
Detección de armas y riesgos
Resúmenes automáticos diarios
IA tipo “chat con video”
🤖 Licencia

MIT License


---