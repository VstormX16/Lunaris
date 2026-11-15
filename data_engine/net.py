"""Resilient HTTP download. TLS verification stays ON; retries + atomic .part files."""
import time
from pathlib import Path

import requests

_HEADERS = {"User-Agent": "lunar-site-analyzer/0.1"}


def remote_size(url, timeout=30):
    r = requests.head(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def download(url, dest, retries=3, timeout=60, chunk=1 << 20):
    """Download url -> dest unless a complete copy exists. Returns dest."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = remote_size(url)
    except requests.RequestException:
        if dest.exists():  # offline or the server is down: a finished download is complete (it is only ever moved in from .part)
            return dest
        raise
    if dest.exists() and dest.stat().st_size == size:
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=_HEADERS, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
            if part.stat().st_size != size:
                raise IOError(f"size mismatch for {url}")
            part.replace(dest)
            return dest
        except (requests.RequestException, IOError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
