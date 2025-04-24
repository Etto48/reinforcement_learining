import sys
import time
import numpy as np
import socket
import argparse as ap
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2

WINDOW = "RGB Array"

def wait():
    cv2.waitKey(1)
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        sys.exit(0)

def connect(host, port) -> socket.socket:
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
            pass
        except OSError:
            ip_index += 1
            if ip_index >= len(addr_info):
                print("Could not connect to any address associated with the host.")
                sys.exit(1)
    print("Server started, waiting for data...")
    return sock

def main(host, port):
    # create a TCP socket
    sock = connect(host, port)
    cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    # handle the closing of the window
    while True:
        # receive data from the client
        data_info = b""
        data_info_len = 4
        while len(data_info) < data_info_len:
            try:
                chunk = sock.recv(data_info_len - len(data_info))
                if not chunk:
                    break
                data_info += chunk
            except BlockingIOError:
                wait()
                pass
        if len(data_info) != 4:
            print("Server disconnected, trying to reconnect...")
            sock = connect(host, port)
            continue
        data_info = int.from_bytes(data_info, byteorder="big")
        if data_info > 1920 * 1080 * 3:
            print(f"Invalid info received. {data_info}")
            break
        # receive the image data
        data = b""
        while len(data) < data_info:
            # receive the data in chunks
            try:
                chunk = sock.recv(data_info - len(data))
                if not chunk:
                    break
                data += chunk
            except BlockingIOError:
                wait()
                pass
        if len(data) != data_info:
            print("Server disconnected, trying to reconnect...")
            sock = connect(host, port)
            continue
        image = np.frombuffer(data, dtype=np.uint8)
        rgb_array = cv2.imdecode(image, cv2.IMREAD_COLOR)
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
        # display the image
        cv2.imshow(WINDOW, rgb_array)
        wait()

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="RGB Array Client")
    parser.add_argument("--host", type=str, default="localhost", required=False, help="Host to connect to")
    parser.add_argument("--port", type=int, default=8048, required=False, help="Port to connect to")
    args = parser.parse_args()
    main(args.host, args.port)