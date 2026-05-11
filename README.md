# CamAiSearch - Sistema de vigilancia inteligente con IA

Backend modular en Python para analítica de video empresarial:

- Analiza videos completos, RTSP/IP/USB y streams en vivo.
- Detecta personas/objetos, hace tracking y reconocimiento facial.
- Genera eventos automáticos con timeline y clips.
- Permite búsqueda semántica por lenguaje natural.
- Expone API FastAPI + WebSocket para dashboard en tiempo real.

## Arquitectura

```text
project/
├── main.py
├── detector/      # YOLO detección
├── tracking/      # Tracking de IDs
├── face/          # InsightFace
├── search/        # Búsqueda semántica
├── embeddings/    # CLIP embeddings texto/imagen
├── llm/           # Descripción de escenas (BLIP) + Whisper opcional
├── events/        # Inferencia de eventos + clips
├── database/      # SQLAlchemy (SQLite/PostgreSQL)
├── api/           # FastAPI + dashboard + WebSocket
├── pipeline/      # Pipeline de análisis offline/live
├── models/        # Schemas API
├── output/        # Frames, clips, DB, grabaciones
├── videos/        # Videos de entrada
└── config/        # Configuración JSON
```

## Requisitos

- Python 3.12+ recomendado (compatible con 3.11 en esta base)
- CUDA opcional para inferencia acelerada
- FFmpeg recomendado para operaciones de video avanzadas

Instalación:

```bash
pip install -r requirements.txt
```

### Google Colab

Para ejecutar en Colab sin depender de CUDA local:

```bash
!git clone https://github.com/CarlosGTI001/CamAiSearch.git
%cd CamAiSearch
!pip install -r requirements-colab.txt
!python main.py --config config/config.colab.json --host 0.0.0.0 --port 8000
```

También tienes un notebook listo para usar: `CamAiSearch_Colab.ipynb`.

El notebook ya incluye una celda para exponer la UI completa con túnel de Cloudflare (TryCloudflare) y acceder al dashboard desde una URL pública.

Luego puedes invocar la API desde otra celda, por ejemplo:

```bash
!curl http://127.0.0.1:8000/events
```

## Configuración

Edita `config/config.json`:

- `cameras`: RTSP/IP/USB (ejemplo `0` para webcam)
- `thresholds`: confianza detector, similitud facial, similitud semántica
- `runtime`: FPS de análisis, buffer, reconexión, paralelismo
- `models`: rutas/nombres YOLO, CLIP, BLIP, Whisper e InsightFace
- `database_url`: `sqlite:///output/camai.db` o `postgresql+psycopg://...`

## Ejecución

```bash
python main.py --config config/config.json --host 0.0.0.0 --port 8000
```

Dashboard:

- `http://localhost:8000/`

## Endpoints principales

- `POST /analyze` analizar archivo de video o stream puntual
- `GET /search?q=persona corriendo` búsqueda inteligente
- `GET /events` timeline de eventos
- `GET /faces` listar personas registradas
- `POST /faces` registrar rostro conocido
- `GET /clips` clips generados
- `GET /alerts` alertas automáticas
- `GET /live-events` eventos en vivo
- `GET /stream/{camera_id}` MJPEG con bounding boxes
- `GET /snapshot?camera_id=...` snapshot inmediato
- `GET /recordings` grabaciones de eventos
- `GET /heatmap/{camera_id}` heatmap de movimiento
- `WS /ws/live-events` stream de eventos para UI

## Ejemplos de consultas semánticas

- `persona con camiseta roja`
- `alguien entrando por la puerta`
- `persona con gorra negra`
- `cuando Carlos apareció`
- `persona corriendo`
- `objeto abandonado`
- `momentos sospechosos`

## Eventos soportados (base)

- `person_detected`
- `person_recognized`
- `person_running`
- `intrusion`
- `line_crossing`
- `abandoned_object`
- `suspicious_activity`

## Pipeline de análisis

1. Captura frame (offline/live).
2. Detección (YOLO) de personas/vehículos/mochilas/cajas/objetos.
3. Tracking por ID (tracker centroidal, extensible a ByteTrack/DeepSORT).
4. Reconocimiento facial (InsightFace, si está instalado).
5. Inferencia de eventos (intrusión, correr, cruce, abandonado, sospechoso).
6. Descripción de escena (BLIP o fallback por reglas).
7. Embedding multimodal (CLIP imagen + texto consulta).
8. Persistencia en DB (eventos, rostros, clips, alertas, metadatos).
9. Búsqueda semántica por similitud de embeddings.

## Escalabilidad y extensibilidad

- Procesamiento paralelo por cámara con workers dedicados.
- Reconexión automática RTSP/IP.
- Buffer circular para generación automática de clips.
- Soporte SQLite y PostgreSQL vía SQLAlchemy.
- Backend listo para dashboard web, timeline visual y búsquedas tipo chat.

## Nota de producción

Para despliegue empresarial:

- Usa modelos YOLO/CLIP/BLIP de mayor capacidad.
- Ajusta umbrales por cámara y escenario.
- Añade Redis + cola de tareas para cargas altas.
- Añade NATS/Kafka para eventos distribuidos.
- Integra notificaciones (Telegram/Discord/email/push) sobre `/alerts`.
