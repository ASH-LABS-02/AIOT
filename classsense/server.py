# classsense/server.py
# Headless output: an MJPEG video stream and a JSON status endpoint.
#
#   http://<pi>:8080/          a page with the live view and the tallies
#   http://<pi>:8080/stream    raw MJPEG, for an <img> tag or VLC
#   http://<pi>:8080/status    JSON, for a dashboard or a polling script
#   http://<pi>:8080/snapshot  a single JPEG
#
# Standard library only. MJPEG is not an efficient codec, but it needs no
# broker, no WebRTC negotiation, no client software and no extra dependency on
# a machine where every dependency costs install time and memory - a browser on
# a phone renders it from a plain <img> tag.
#
# The Pi is assumed to be on a trusted local network: there is no authentication
# and the stream shows a live camera feed of a classroom. Do not port-forward it.

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "classsense-frame"

PAGE = """<!doctype html>
<title>ClassSense</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#111; color:#eee;
         font:14px system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:10px 14px; background:#1b1b1b; border-bottom:1px solid #2c2c2c; }
  h1 { margin:0; font-size:15px; font-weight:600; letter-spacing:.2px; }
  #wrap { display:flex; flex-wrap:wrap; gap:14px; padding:14px; }
  img { max-width:100%; border-radius:6px; background:#000; flex:1 1 480px; }
  #side { flex:0 0 230px; }
  .row { display:flex; justify-content:space-between; padding:5px 0;
         border-bottom:1px solid #262626; }
  .k { color:#9aa0a6; }
  .v { font-variant-numeric:tabular-nums; }
  .att { color:#31c48d; } .slp { color:#f6a609; }
  .dis { color:#f05252; } .unk { color:#9aa0a6; }
  #note { padding:0 14px 14px; color:#9aa0a6; font-size:12px; max-width:60ch; }
</style>
<header><h1>ClassSense</h1></header>
<div id="wrap">
  <img src="/stream" alt="live view">
  <div id="side"></div>
</div>
<div id="note"></div>
<script>
async function tick() {
  try {
    const s = await (await fetch('/status')).json();
    const c = s.counts || {};
    document.getElementById('side').innerHTML = [
      ['Students', s.students, ''],
      ['Attentive', c.Attentive|0, 'att'],
      ['Sleepy', c.Sleepy|0, 'slp'],
      ['Distracted', c.Distracted|0, 'dis'],
      ['Unreadable', c.Unknown|0, 'unk'],
      ['Unmonitored', c.Unmonitored|0, 'unk'],
      ['Engagement', s.engagement_pct + '%', ''],
      ['Analysis', s.analysis_per_sec + '/s', ''],
      ['Cycle', s.cycle_ms + ' ms', ''],
    ].map(([k,v,cls]) =>
      `<div class="row"><span class="k">${k}</span>`+
      `<span class="v ${cls}">${v}</span></div>`).join('');
    document.getElementById('note').textContent = s.note || '';
  } catch (e) { /* server restarting; next tick will retry */ }
}
tick(); setInterval(tick, 1000);
</script>
"""


class _Handler(BaseHTTPRequestHandler):
    provider = None           # set by serve()

    # Quieten the default per-request logging: an MJPEG client reconnecting
    # would otherwise bury anything useful the pipeline prints.
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, content_type, body=None, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        if body is not None:
            self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body is not None:
            self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/status":
            body = json.dumps(self.provider.status(), indent=2).encode("utf-8")
            self._send(200, "application/json", body,
                       {"Cache-Control": "no-store"})
        elif path == "/snapshot":
            jpeg = self.provider.jpeg()
            if jpeg is None:
                self._send(503, "text/plain", b"no frame yet")
            else:
                self._send(200, "image/jpeg", jpeg, {"Cache-Control": "no-store"})
        elif path == "/stream":
            self._stream()
        else:
            self._send(404, "text/plain", b"not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        try:
            while True:
                jpeg = self.provider.jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                # Cap the stream rate. The analysis runs a few times a second
                # and the Pi should spend its cores on that, not on encoding
                # JPEGs nobody can perceive the difference in.
                time.sleep(self.provider.stream_interval)
        except (BrokenPipeError, ConnectionResetError):
            pass          # viewer closed the tab; entirely normal


class StreamProvider:
    """
    Holds the newest rendered frame and the newest status, for the handlers.

    Encoding happens here, once per new frame, rather than once per connected
    viewer - so a second person opening the page costs nothing but bandwidth.
    """

    def __init__(self, jpeg_quality=70, stream_fps=8):
        self._frame = None
        self._jpeg = None
        self._jpeg_stamp = -1.0
        self._stamp = 0.0
        self._status = {}
        self._lock = threading.Lock()
        self.jpeg_quality = jpeg_quality
        self.stream_interval = 1.0 / max(1, stream_fps)

    def publish(self, frame, status):
        with self._lock:
            self._frame = frame
            self._status = status
            self._stamp = time.time()

    def jpeg(self):
        with self._lock:
            frame = self._frame
            stamp = self._stamp
            if frame is None:
                return None
            if self._jpeg is not None and stamp == self._jpeg_stamp:
                return self._jpeg          # already encoded this frame
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return None
        data = buf.tobytes()
        with self._lock:
            self._jpeg = data
            self._jpeg_stamp = stamp
        return data

    def status(self):
        with self._lock:
            return dict(self._status)


def serve(provider, port=8080, host="0.0.0.0"):
    """Start the HTTP server on a daemon thread and return it."""
    handler = type("Handler", (_Handler,), {"provider": provider})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
