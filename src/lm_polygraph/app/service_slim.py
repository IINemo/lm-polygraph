

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-path", type=str,
                        default=os.path.expanduser('~') + '/.cache')

    args = parser.parse_args()
    cache_path = args.cache_path
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'

    static_folder = 'client'
    app = Flask(__name__, static_folder=static_folder)


    @app.route('/')
    def serve_index():
        return send_from_directory(os.path.join(app.root_path, static_folder), 'index.html')


    @app.route('/<path:filename>')
    def serve_static(filename):
        return send_from_directory(os.path.join(app.root_path, static_folder), filename)


    @app.route('/get-prompt-result', methods=['GET', 'POST'])
    def generate():
        data = request.get_json()
        return _generate_response(data)

    app.run(host='0.0.0.0', port=args.port, debug=True)
