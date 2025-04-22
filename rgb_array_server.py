import socket

import numpy as np


class RgbArrayServer:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", 8048))
        self.socket.settimeout(0.1)
        self.socket.listen(1)
        self.client_socket = None

    def send(self, data: np.ndarray):
        if self.client_socket is None:
            try:
                self.client_socket, _ = self.socket.accept()
            except socket.timeout:
                self.client_socket = None
                return
        shape = data.shape
        shape_bytes = np.array(shape, dtype=np.int32)
        try:
            self.client_socket.sendall(shape_bytes.tobytes())
            self.client_socket.sendall(data.tobytes())
        except (BrokenPipeError, ConnectionResetError):
            self.client_socket.close()
            self.client_socket = None
    def __del__(self):
        if self.client_socket:
            self.client_socket.close()
        self.socket.close()