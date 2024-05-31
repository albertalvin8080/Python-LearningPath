from http.server import HTTPServer, BaseHTTPRequestHandler
# from overrides import override # requires pip install overrides
import traceback

class MyHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("ContentType", "text/html")
        self.end_headers() # This need to come before `self.wfile.write()``

        self.wfile.write(bytes("<html><body><h1>Hello, Lain!</h1></body></html>", "utf-8"))
    
if __name__ == "__main__":
    HOST = 'localhost'
    PORT = 8080
    # HOST + PORT = Address
    server = HTTPServer((HOST, PORT), MyHTTPHandler)

    try:
        print(f"Starting server at: {HOST}:{PORT}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrupt catched, shutting down...")
        server.server_close() # Conflicts with ctrl+c termination signal
        # traceback.print_exc()
    
