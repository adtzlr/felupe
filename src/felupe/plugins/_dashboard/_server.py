# SPDX-License-Identifier: GPL-3.0-or-later

"""A small HTTP server which streams the events of a job as Server-Sent Events."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Broadcaster:
    """A thread-safe log of events for all connected clients.

    Persistent events are kept in a history and every client receives all of them
    (also clients which connect late). A transient event, e.g. the field values of a
    non-converged iteration, is kept in a single slot and replaced by the next one.
    Slow clients only receive the latest transient event.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self.history = []
        self._transient = (0, None)
        self._version = 0
        self._generation = 0

    def publish(self, message, transient=False):
        with self._cond:
            if transient:
                self._version += 1
                self._transient = (self._version, message)
            else:
                self.history.append(message)
            self._cond.notify_all()

    def reset(self):
        "Clear the history, e.g. on a repeated evaluation of a job."
        with self._cond:
            self.history = []
            self._transient = (0, None)
            self._generation += 1
            self._cond.notify_all()

    def stream(self, write, heartbeat=15.0):
        "Write all events to a client until the connection is closed."
        cursor = 0
        version = 0
        generation = self._generation

        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: (
                        len(self.history) > cursor
                        or self._transient[0] > version
                        or self._generation != generation
                    ),
                    timeout=heartbeat,
                )
                if self._generation != generation:
                    generation = self._generation
                    cursor = 0
                    version = 0
                    pending = ['{"type":"reset"}']
                else:
                    pending = []
                pending += self.history[cursor:]
                cursor = len(self.history)
                if self._transient[0] > version:
                    version, message = self._transient
                    pending.append(message)

            if pending:
                write("".join(f"data: {m}\n\n" for m in pending))
            else:
                write(": keep-alive\n\n")


def serve(page, broadcaster, host="127.0.0.1", port=0):
    """Start the server in a daemon thread and return it.

    Parameters
    ----------
    page : callable
        A callable which returns the HTML page as bytes.
    broadcaster : Broadcaster
        The log of events.
    host : str, optional
        The host of the server (default is "127.0.0.1").
    port : int, optional
        The port of the server. With 0, a free port is chosen (default is 0).
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]

            if path in ("/", "/index.html"):
                body = page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                def write(text):
                    self.wfile.write(text.encode("utf-8"))
                    self.wfile.flush()

                try:
                    broadcaster.stream(write)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            else:
                self.send_error(404)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
