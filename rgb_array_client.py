import sys
import time
import numpy as np
import socket
import argparse as ap
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2

WINDOW = "RGB Array"
LAST_IMAGE = np.zeros((400, 600, 3), dtype=np.uint8)

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

def paused():
    img = LAST_IMAGE // 2 + monochrome_image("white") // 2
    cv2.putText(img, "Waiting...", (50, LAST_IMAGE.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow(WINDOW, img)

def disconnected_image():
    img = monochrome_image("blue")
    cv2.putText(img, "Connecting...", (50, LAST_IMAGE.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(WINDOW, img)

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
    return sock

def main(host, port):
    global LAST_IMAGE
    # create a TCP socket
    cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    sock = connect(host, port)
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
        # avoid ddos
        if data_info > 1920 * 1080 * 3:
            print(f"Invalid info received. {data_info}")
            break
        # check if the server requested to clear the image
        match data_info:
            case 0:
                paused()
                continue
            case i if i < 128:
                # reserved for future use
                pass
            case _:
                pass
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
        LAST_IMAGE = rgb_array
        # display the image
        cv2.imshow(WINDOW, rgb_array)
        wait()

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="RGB Array Client")
    parser.add_argument("--host", type=str, default="localhost", required=False, help="Host to connect to")
    parser.add_argument("--port", type=int, default=8048, required=False, help="Port to connect to")
    args = parser.parse_args()
    main(args.host, args.port)