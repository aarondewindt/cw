from random import randint
import psutil
import socket


# TODO: Write unittests for net_tools.


def get_available_port():
    """
    Returns the port number of an available port.
    """
    first_port = randint(1025, 65535)
    port = first_port
    open_ports = []
    for sconn in psutil.net_connections():
        open_ports.append(sconn.laddr[1])
    while (port in open_ports) and (port-first_port < 100):
        port += 1
    if port == (first_port+100):
        raise RuntimeError("Unable to find available port.")
    else:
        return port


def has_internet(host="8.8.8.8", port=53, timeout=3):
    """
    Check whether the computer is connected to the internet by trying to
    connect to an online server. By default this is '8.8.8.8', the google
    public DNS.

    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except Exception as ex:
        print(ex)
        return False

