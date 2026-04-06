from prometheus_client import start_http_server
import socket

def start_metrics():
    try:
        sock = socket.socket()
        sock.bind(("0.0.0.0", 8000))
        sock.close()
        start_http_server(8000)
    except:
        pass

if "metrics_started" not in st.session_state:
    start_metrics()
    st.session_state["metrics_started"] = True