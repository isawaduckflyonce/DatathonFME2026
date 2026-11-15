"""Control the Arduino UNO Q 8x13 LED matrix via /var/run/arduino-router.sock."""
import os
import socket
import threading

try:
    from arduino.app_utils import Bridge
except ImportError:
    Bridge = None

ROUTER_SOCK = "/var/run/arduino-router.sock"
CLEAR = 0
TICK = 1
CROSS = 2


def _rpc_show_result(state, timeout=2.0):
    """Send MessagePack-RPC REQUEST [0, 1, "show_result", [state]] to arduino-router."""
    payload = b"\x94\x00\x01\xabshow_result\x91" + bytes([int(state) & 0x7F])
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(ROUTER_SOCK)
        s.sendall(payload)
        return s.recv(64)


def _send(state):
    try:
        if os.path.exists(ROUTER_SOCK):
            _rpc_show_result(state)
        elif Bridge is not None:
            Bridge.call("show_result", int(state), timeout=2)
    except Exception:
        pass


def show_matrix(state):
    """Fire-and-forget update of the 8x13 LED matrix (0=clear, 1=tick, 2=cross)."""
    threading.Thread(target=_send, args=(state,), daemon=True).start()


def show_verdict(correct):
    show_matrix(TICK if correct else CROSS)


def clear_matrix():
    show_matrix(CLEAR)