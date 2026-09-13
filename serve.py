"""Cross-platform production WSGI entry point; no Flask development server."""
import logging
import os

from service import create_app


def main():
    from waitress import serve
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.extensions["warmup"]()
    host = "0.0.0.0" if app.config["ENVIRONMENT"] == "production" else "127.0.0.1"
    serve(app, host=host, port=int(os.getenv("PORT", "8080")), threads=4,
          connection_limit=32, channel_timeout=60,
          max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
          expose_tracebacks=False)


if __name__ == "__main__":
    main()
