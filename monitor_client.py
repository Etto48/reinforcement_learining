import sys
import time
from typing import Optional
import numpy as np
import socket
import argparse as ap
import json
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2

WINDOW = "Monitor"
LAST_IMAGE = np.zeros((400, 600, 3), dtype=np.uint8)
LAST_INFO = {}
PAUSED = False

def wait():
    cv2.waitKey(1)
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        sys.exit(0)

def monochrome_image(color: str):
    match color:
        case "red":
            c = (0, 0, 255)
        case "green":
            c = (0, 255, 0)
        case "blue":
            c = (255, 0, 0)
        case "yellow":
            c = (0, 255, 255)
        case "cyan":
            c = (255, 255, 0)
        case "magenta":
            c = (255, 0, 255)
        case "white":
            c = (255, 255, 255)
        case "black":
            c = (0, 0, 0)
        case "gray":
            c = (128, 128, 128)
        case "dark_gray":
            c = (64, 64, 64)
        case _:
            raise ValueError(f"Invalid color: {color}")
    img = np.zeros(LAST_IMAGE.shape, dtype=np.uint8)
    img[:] = c
    return img

def get_pause_image():
    img = cv2.addWeighted(LAST_IMAGE, 0.5, monochrome_image("white"), 0.5, 0)
    cv2.GaussianBlur(img, (11, 11), 0, img)
    cv2.putText(img, "Waiting...", (50, LAST_IMAGE.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

def render_image():
    global LAST_IMAGE, PAUSED, LAST_INFO
    if PAUSED:
        img = get_pause_image()
    else:
        img = LAST_IMAGE.copy()
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    offset = 0
    for key, value in LAST_INFO.items():
        if isinstance(value, int):
            text = f"{key}: {value}"
        elif isinstance(value, float):
            text = f"{key}: {value:.2f}"
        elif isinstance(value, str):
            text = f"{key}: {value}"
        else:
            raise ValueError(f"Invalid value type: {type(value)}")
        cv2.putText(img, text, (50, offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bg_color, 5)
        cv2.putText(img, text, (50, offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        offset += 30
    cv2.imshow(WINDOW, img)

def paused():
    global PAUSED
    PAUSED = True
    render_image()

def show_info(info: dict[str, int | float | str]):
    global LAST_IMAGE, LAST_INFO, PAUSED
    LAST_INFO = info
    render_image()

def show_image(image: np.ndarray):
    global LAST_IMAGE, PAUSED, LAST_INFO
    PAUSED = False
    LAST_IMAGE = image
    render_image()

def disconnected_image():
    img = monochrome_image("blue")
    cv2.putText(img, "Connecting...", (50, LAST_IMAGE.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(WINDOW, img)

def deserialize_dict(data: bytes) -> dict:
    data = data.decode("utf-8")
    data = json.loads(data)
    return data

def deserialize_image(data: bytes) -> np.ndarray:
    image = np.frombuffer(data, dtype=np.uint8)
    rgb_array = cv2.imdecode(image, cv2.IMREAD_COLOR)
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
    return rgb_array

class DisconnectedException(Exception):
    pass

def recv(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        try:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise DisconnectedException
            data += chunk
        except BlockingIOError:
            wait()
            pass
    return data

def connect(host, port) -> socket.socket:
    disconnected_image()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    addr_info = socket.getaddrinfo(host, None)
    print(f"Connecting to {host}:{port}")
    ip_index = 0
    while True:
        try:
            ip_address = addr_info[ip_index][4][0]
            server_address = (ip_address, port)
            sock.connect(server_address)
            break
        except (BlockingIOError, ConnectionRefusedError):
            time.sleep(0.1)
            wait()
            pass
        except OSError:
            ip_index += 1
            if ip_index >= len(addr_info):
                print("Could not connect to any address associated with the host.")
                sys.exit(1)
    print("Server started, waiting for data...")
    paused()
    return sock

def main(host, port):
    global LAST_IMAGE
    # create a TCP socket
    cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    sock = connect(host, port)
    # handle the closing of the window
    while True:
        try:
            # receive data from the client
            data_info = recv(sock, 4)
            data_info = int.from_bytes(data_info, byteorder="big")
            # avoid ddos
            if data_info > 1920 * 1080 * 3:
                print(f"Invalid info received. {data_info}")
                break
            # check if the server requested to clear the image
            match data_info:
                case 0:
                    # pause request
                    paused()
                    continue
                case 1:
                    # training information packet
                    info_len = recv(sock, 4)
                    info_len = int.from_bytes(info_len, byteorder="big")
                    if info_len > 1024 * 10:
                        print(f"Invalid info length received. {info_len}")
                        break
                    data = recv(sock, info_len)
                    data = deserialize_dict(data)
                    show_info(data)
                case i if i < 128:
                    # reserved for future use
                    pass
                case _:
                    data = recv(sock, data_info)
                    image = deserialize_image(data)
                    show_image(image)
            wait()
        except DisconnectedException:
            print("Disconnected from server.")
            sock.close()
            sock = connect(host, port)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Monitor Client")
    parser.add_argument("--host", type=str, default="localhost", required=False, help="Host to connect to")
    parser.add_argument("--port", type=int, default=8048, required=False, help="Port to connect to")
    args = parser.parse_args()
    main(args.host, args.port)