import socket

import numpy as np
import cv2


class RgbArrayServer:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", 8048))
        self.socket.setblocking(False)
        self.socket.listen(1)
        self.client_socket = None

    def serialize(self, data: np.ndarray):
        _, frame = cv2.imencode(".jpg", data)
        frame = frame.tobytes()
        data = len(frame).to_bytes(4, byteorder="big") + frame
        return data

    def is_connected(self):
        if self.client_socket is None:
            try:
                self.client_socket, _ = self.socket.accept()
                return True
            except BlockingIOError:
                self.client_socket = None
                return False
        else:
            return True

    def send(self, data: np.ndarray):
        if not self.is_connected():
            return
        try:
            data = self.serialize(data)
            self.client_socket.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            self.client_socket.close()
            self.client_socket = None
    def __del__(self):
        if self.client_socket:
            self.client_socket.close()
        self.socket.close()