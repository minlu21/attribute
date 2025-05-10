import http.server
import fire


class RemoveCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()


@fire.Fire
def main(port: int = 8000):
    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, RemoveCacheHandler)
    print(f"Serving on port {port}...")
    httpd.serve_forever()
