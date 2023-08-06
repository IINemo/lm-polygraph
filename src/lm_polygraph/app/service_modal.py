from pathlib import Path
from modal import Image, Stub, wsgi_app
from flask import Flask, request, abort, send_from_directory, render_template
import os


static_folder = 'client'
static_path = Path(__file__).parent / static_folder

polygraph_image = Image.from_dockerhub("mephodybro/polygraph_demo_base")
# dockerfile_path = Path(__file__).parent.parent.parent.parent / 'Dockerfile'
# polygraph_image = Image.from_dockerfile(dockerfile_path)


stub = Stub(
    "demo_polygraph",
    image=polygraph_image,
)


@stub.function(gpu='any')
@wsgi_app()
def flask_app():
    app = Flask(__name__, static_folder=static_path)

    @app.get("/")
    def home():
        return f"{os.listdir('.')}\n{static_path} "

    # @app.route('/')
    # def serve_index():
    #     return send_from_directory(static_path, 'index.html')
    #
    # @app.route('/<path:filename>')
    # def serve_static(filename):
    #     return send_from_directory(static_path, filename)
    #
    # @app.get("/debug")
    # def home():
    #     return f"{os.listdir('.')}\n{static_path} "

    return app