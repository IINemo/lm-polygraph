import json
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Optional, Dict, Any
import time


class MockOpenAIHandler(BaseHTTPRequestHandler):
    """Handler for mock OpenAI API requests"""

    # Class-level configuration
    fixed_response: str = "This is a mock response"
    error_rate: float = 0.0  # Probability of returning an error response
    response_delay: float = 0.0  # Delay in seconds before responding

    def do_POST(self):
        """Handle POST requests to the mock API"""
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Only /v1/chat/completions endpoint is supported")
            return

        # Read and parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(content_length).decode("utf-8")

        try:
            request_data = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON in request body")
            return

        # Validate request format
        if not self._validate_request(request_data):
            self.send_error(400, "Invalid request format")
            return

        # Prepare response
        response = {
            "id": f"mock-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request_data.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.fixed_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(self.fixed_response.split()),
                "total_tokens": 10 + len(self.fixed_response.split()),
            },
        }

        # Send response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def _validate_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate the incoming request format"""
        # Check for required fields
        if "messages" not in request_data:
            return False

        # Validate messages format
        messages = request_data["messages"]
        if not isinstance(messages, list) or not messages:
            return False

        # Validate each message
        for message in messages:
            if not isinstance(message, dict):
                return False
            if "role" not in message or "content" not in message:
                return False

        return True

    def log_message(self, format: str, *args) -> None:
        """Override to disable request logging"""
        pass


class MockOpenAIServer:
    """
    Mock OpenAI API server for testing purposes

    Usage:
        server = MockOpenAIServer(port=8080)
        server.start()
        # Run your tests
        server.stop()
    """

    def __init__(self, host: str = "localhost", port: int = 8080, max_retries: int = 5):
        self.host = host
        self.initial_port = port
        self.port = port
        self.max_retries = max_retries
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None

    def _find_available_port(self) -> int:
        """Find an available port starting from the initial port"""
        current_port = self.initial_port

        for _ in range(self.max_retries):
            try:
                # Test if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, current_port))
                return current_port
            except OSError:
                current_port += 1

        raise RuntimeError(
            f"Could not find an available port after {self.max_retries} attempts"
        )

    def start(self) -> None:
        """Start the mock server in a separate thread"""
        if self.server:
            raise RuntimeError("Server is already running")

        # Find available port
        self.port = self._find_available_port()

        try:
            self.server = HTTPServer((self.host, self.port), MockOpenAIHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()

            # Wait briefly to ensure server starts
            time.sleep(0.1)
        except Exception as e:
            self.stop()  # Clean up if startup fails
            raise RuntimeError(f"Failed to start server: {str(e)}")

    def stop(self) -> None:
        """Stop the mock server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.server_thread = None

    def __enter__(self):
        """Context manager support"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()

    @property
    def base_url(self) -> str:
        """Get the base URL for the server"""
        return f"http://{self.host}:{self.port}"

    @classmethod
    def configure(
        cls,
        fixed_response: Optional[str] = None,
        error_rate: Optional[float] = None,
        response_delay: Optional[float] = None,
    ) -> None:
        """Configure the mock server's behavior"""
        if fixed_response is not None:
            MockOpenAIHandler.fixed_response = fixed_response
        if error_rate is not None:
            MockOpenAIHandler.error_rate = error_rate
        if response_delay is not None:
            MockOpenAIHandler.response_delay = response_delay
