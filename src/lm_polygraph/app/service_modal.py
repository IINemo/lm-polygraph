from pathlib import Path
from modal import Image, Stub, wsgi_app
from flask import Flask, request, abort, send_from_directory, render_template
import os


static_folder = 'client'
static_path = Path(__file__).parent / static_folder

stub = Stub(
    "demo_polygraph",
    image=Image.debian_slim().pip_install("flask").copy_local_dir(static_path, '/root/lm_polygraph/app/client'),
)



@stub.function()
@wsgi_app()
def flask_app():
    app = Flask(__name__, static_folder=static_path)
    @app.get("/")
    def home():
        return f"{os.listdir('.')}\n{static_path} "

    # @app.route('/')
    # def serve_index():
    #     return send_from_directory(os.path.join(app.root_path, static_folder), 'index.html')

    @app.route('/<path:filename>')
    def serve_static(filename):
        return send_from_directory(static_path, filename)

    # @web_app.get("/")
    # def home():
    #     return "Hello Flask World!"
    #
    # @web_app.post("/foo")
    # def foo():
    #     return request.json

    return app