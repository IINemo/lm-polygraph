import os
from pathlib import Path
from modal import Image, Stub, wsgi_app
from flask import Flask, send_from_directory, request
from lm_polygraph.app.service_helpers import _generate_response


# static_folder = 'client'
# static_path = Path(__file__).parent / static_folder
static_path = '/app/src/lm_polygraph/app/client'

polygraph_image = Image.from_dockerhub("mephodybro/polygraph_demo:0.0.9")


stub = Stub(
    "demo_polygraph",
    image=polygraph_image,
)


@stub.function(gpu='any')
@wsgi_app()
def flask_app():
    app = Flask(__name__, static_folder=static_path)
    device = 'cuda'

    @app.get("/debug")
    def home():
        result = '\n'.join([
            os.getcwd(),
            str(os.liktdir('.')),
            str(os.listdir('./src/lm_polygraph/app')),
            str(static_path),
            # str(os.listdir('/app')),
            # str(os.listdir(static_path))
        ])
        return result
        # return f"{os.getcwd()}\n{os.listdir('.')}\n{static_path} "

    @app.route('/')
    def serve_index():
        return send_from_directory(static_path, 'index.html')

    @app.route('/<path:filename>')
    def serve_static(filename):
        return send_from_directory(static_path, filename)

    @app.route('/get-prompt-result', methods=['GET', 'POST'])
    def generate():
        data = request.get_json()
        return _generate_response(data)

    return app