"""Minimal alert receiver for locally verifying the Prometheus -> Alertmanager pipeline.

Not part of the app's runtime — this stands in for Slack/PagerDuty/email so the
alert pipeline can be proven end-to-end without a real integration configured.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG_PATH = "webhook_received.jsonl"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        payload = json.loads(body)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
        for alert in payload.get("alerts", []):
            print(f"[{alert['status'].upper()}] {alert['labels'].get('alertname')}: {alert['annotations'].get('summary')}")
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 9099), Handler)
    print("Webhook receiver listening on :9099")
    server.serve_forever()
