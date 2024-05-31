from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# http://localhost:8080/home?name=Pedro&age=999&name=lucas&age=-999
class MyHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print("##############################################")
        unparsed_path = self.path
        parsed_path = urlparse(unparsed_path)
        params = parse_qs(parsed_path.query)

        print("Path:", unparsed_path)
        print("Parsed Path:", parsed_path.path)
        print("Parameters:", params)

        if parsed_path.path == "/home":
            self.handle_home()
        else:
            self.handle_404()

    def handle_home(self):
        self.send_response(200)
        self.send_header("ContentType", "text/html;charset=utf-8")
        self.end_headers()
        with open("./static/html/home.html", "+br") as f:
            page = f.read()
        self.wfile.write(page)

    def handle_404(self):
        self.send_response(404)
        self.send_header("ContentType", "text/plain")
        self.end_headers()
        self.wfile.write(bytes("Not Found", "utf-8"))


if __name__ == "__main__":
    address = ("localhost", 8080)
    server = HTTPServer(address, MyHTTPHandler)

    try:
        print(f"Starting server at: {address[0]}:{address[1]}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
