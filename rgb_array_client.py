import time
import numpy as np
import matplotlib.pyplot as plt
import socket

def main():
    # create a TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("localhost", 8048)
    for i in range(100):
        try:
            sock.connect(server_address)
            break
        except ConnectionRefusedError:
            print(f"Connection attempt {i + 1} failed. Retrying...")
            time.sleep(1)
            pass
    print("Server started, waiting for data...")
    plt.figure()
    plt.ion()
    plt.show()
    while True:
        # receive data from the client
        shape_data = sock.recv(4 * 3)
        if not shape_data:
            print("No data info received from server.")
            break
        # unpack the shape
        size_x, size_y, size_z = np.frombuffer(shape_data, dtype=np.int32)
        if size_z != 3 or size_x > 1920 or size_y > 1080:
            print("Invalid shape received.")
            break
        # receive the image data
        data = b""
        while len(data) < size_x * size_y * size_z:
            # receive the data in chunks
            chunk = sock.recv(size_x * size_y * size_z - len(data))
            if not chunk:
                print("No image data received.")
                break
            data += chunk
        # unpickle the data
        rgb_array = np.frombuffer(data, dtype=np.uint8)
        rgb_array = rgb_array.reshape((size_x, size_y, size_z))
        # display the image
        plt.imshow(rgb_array)
        plt.axis('off')
        plt.pause(0.001)

if __name__ == "__main__":
    main()