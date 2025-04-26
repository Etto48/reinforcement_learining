import socket
import time

import numpy as np
import json
import cv2


class MonitorServer:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", 8048))
        self.socket.setblocking(False)
        self.socket.listen(1)
        self.client_socket = None
        self.is_connected()

    def __del__(self):
        if self.client_socket:
            self.client_socket.close()
        self.socket.close()

    def _serialize_image(self, data: np.ndarray):
        _, frame = cv2.imencode(".jpg", data)
        frame = frame.tobytes()
        data = len(frame).to_bytes(4, byteorder="big") + frame
        return data
    
    def _serialize_dictionary(self, data: dict[str, int | float | str]):
        for key, value in data.items():
            if isinstance(value, np.float32):
                data[key] = float(value)
        data = json.dumps(data).encode("utf-8")
        data = (1).to_bytes(4, byteorder="big") + len(data).to_bytes(4, byteorder="big") + data
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
        data = self._serialize_image(data)
        try:
            self.client_socket.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            self.client_socket.close()
            self.client_socket = None

    def send_paused(self):
        if not self.is_connected():
            return
        try:
            self.client_socket.sendall((0).to_bytes(4, byteorder="big"))
        except (BrokenPipeError, ConnectionResetError):
            self.client_socket.close()
            self.client_socket = None

    def send_info(self, info: dict[str, int | float | str]):
        if not self.is_connected():
            return
        info = self._serialize_dictionary(info)
        try:
            self.client_socket.sendall(info)
        except (BrokenPipeError, ConnectionResetError):
            self.client_socket.close()
            self.client_socket = None

def demo():
    server = MonitorServer()
    i = 0
    paused = False
    while True:
        if (i // 10) % 2 == 0:
            img = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
            server.send(img)
            paused = False
        elif not paused:
            server.send_paused()
            paused = True

        server.send_info({"step": i, "dummy": "info", "reward": np.random.rand()})
        time.sleep(0.1)
        i += 1

if __name__ == "__main__":
    demo()