from __future__ import annotations

import argparse

import uvicorn

from api.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CamAiSearch - Video Intelligence API")
    parser.add_argument("--config", default="config\\config.json", help="Ruta al archivo de configuración JSON")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
