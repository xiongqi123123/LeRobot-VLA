import socket
import numpy as np
import zlib
import json
import base64
import time
from typing import Any
import torch

class NumpyEncoder(json.JSONEncoder):
    """Enhanced json encoder for numpy types and PyTorch tensors with array reconstruction info"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        elif torch is not None and isinstance(obj, torch.Tensor):
            # 将 PyTorch Tensor 转换为 numpy 数组
            numpy_array = obj.cpu().detach().numpy()
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(numpy_array.tobytes()).decode('ascii'),
                'dtype': str(numpy_array.dtype),
                'shape': numpy_array.shape
            }
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)

def json_to_numpy(json_str: str) -> Any:
    def hook(dct):
        if '__numpy_array__' in dct:
            data = base64.b64decode(dct['data'])
            return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    return json.loads(json_str, object_hook=hook)

class CommonUtils:
    @staticmethod
    def serialize(data: Any) -> bytes:
        return zlib.compress(numpy_to_json(data).encode('utf-8'))

    @staticmethod
    def deserialize(data: bytes) -> Any:
        return json_to_numpy(zlib.decompress(data).decode('utf-8'))

def send_all(sock, payload):
    sock.sendall(len(payload).to_bytes(8, 'big') + payload)

def recv_all(sock) :
    length_bytes = sock.recv(8)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, 'big')
    buf = b''
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

class ServerClient:
    def __init__(self, host='localhost', port=5000, is_server=True):
        self.host, self.port, self.is_server = host, port, is_server
        self.utils = CommonUtils()
        self._connect()

    def _connect(self):
        if self.is_server:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind((self.host, self.port))
            self.sock.listen(1)
            print(f"[ServerClient] Listening on {self.host}:{self.port}")
            self.conn, addr = self.sock.accept()
            print(f"[ServerClient] Connected by {addr}")
        else:
            while True:
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.connect((self.host, self.port))
                    self.conn = self.sock
                    print(f"[ServerClient] Connected to {self.host}:{self.port}")
                    break
                except (ConnectionRefusedError, OSError):
                    print("[ServerClient] Waiting for server...")
                    time.sleep(2)

    def send(self, data):
        payload = self.utils.serialize(data)
        try:
            send_all(self.conn, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[ServerClient] Connection lost. Reconnecting...")
            self._connect()
            send_all(self.conn, payload)

    def receive(self):
        try:
            buf = recv_all(self.conn)
            return self.utils.deserialize(buf) if buf else None
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[ServerClient] Connection lost. Reconnecting...")
            self._connect()
            return None

    def close(self):
        self.conn.close()
        self.sock.close()

class Client:
    def __init__(self, host='127.0.0.1', port=5000):
        self.host, self.port = host, port
        self.utils = CommonUtils()
        self.connect()

    def connect(self):
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print(f"[Client] Connected to {self.host}:{self.port}")
                break
            except (ConnectionRefusedError, OSError):
                print("[Client] Waiting for server...")
                time.sleep(2)

    def send(self, data):
        payload = self.utils.serialize(data)
        try:
            send_all(self.sock, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[Client] Connection lost. Reconnecting...")
            self.connect()
            send_all(self.sock, payload)

    def receive(self):
        try:
            buf = recv_all(self.sock)
            return self.utils.deserialize(buf) if buf else None
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[Client] Connection lost. Reconnecting...")
            self.connect()
            return None

    def close(self):
        self.sock.close()
        print("[Client] Closed.")

class Server:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host, self.port = host, port
        self.utils = CommonUtils()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[Server] Listening on {self.host}:{self.port}")
        self._wait_client()

    def _wait_client(self):
        print("[Server] Waiting for client...")
        self.conn, addr = self.sock.accept()
        print(f"[Server] Connected by {addr}")

    def send(self, data: Any):
        payload = self.utils.serialize(data)
        try:
            send_all(self.conn, payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[Server] Client disconnected. Waiting new client...")
            self._wait_client()
            send_all(self.conn, payload)

    def receive(self):
        try:
            buf = recv_all(self.conn)
            return self.utils.deserialize(buf) if buf else None
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[Server] Client disconnected. Waiting new client...")
            self._wait_client()
            return None

    def close(self):
        self.conn.close()
        self.sock.close()
        print("[Server] Closed.")
