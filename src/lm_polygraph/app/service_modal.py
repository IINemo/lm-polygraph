import os
from modal import Image, Stub, wsgi_app
from flask import Flask, send_from_directory, request
from lm_polygraph.app.service_helpers import Responder


# PRELOADED = ['databricks/dolly-v2-7b', 'bigscience/bloomz-3b', 'NousResearch/Llama-2-7b-chat-hf']
PRELOADED = []


def preload():
    from huggingface_hub import snapshot_download

    for model_name in PRELOADED:
        snapshot_download(model_name)


static_path = "/app/src/lm_polygraph/app/client"
polygraph_image = Image.from_dockerhub("mephodybro/polygraph_demo:0.0.18").run_function(
    preload
)


stub = Stub(
    "demo",
    image=polygraph_image,
)


@stub.function(gpu="A100", container_idle_timeout=300, timeout=5 * 60)
@wsgi_app()
def polygraph():
    app = Flask(__name__, static_folder=static_path)

    @app.route("/")
    def serve_index():
        return send_from_directory(static_path, "index.html")

    @app.route("/<path:filename>")
    def serve_static(filename):
        return send_from_directory(static_path, filename)

    @app.route("/get-prompt-result", methods=["GET", "POST"])
    def generate():
        data = request.get_json()
        cache_path = os.path.expanduser("~") + "/.cache"
        device = "cuda"
        responder = Responder(cache_path, device)

        return responder._generate_response(data)

    return app
