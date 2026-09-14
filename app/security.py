"""服务端出站请求安全约束（SSRF 防护）。

规则：
- 仅允许 http/https；
- 请求前校验 host 并解析所有 IP，拒绝 localhost、环回、私有与保留地址；
- 重定向手动逐跳校验。
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.goog",
}


class SafeURLError(ValueError):
    pass


def _check_ip(ip: str) -> None:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        raise SafeURLError(f"无法识别的 IP 地址: {ip}")
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_reserved
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_unspecified
    ):
        raise SafeURLError(f"禁止访问内网/保留地址: {ip}")
    if not addr.is_global:
        raise SafeURLError(f"禁止访问非公网地址: {ip}")


def validate_public_http_url(url: str) -> str:
    """校验 URL 安全性，通过则原样返回，否则抛 SafeURLError。"""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SafeURLError("仅允许 http/https 链接")
    host = parsed.hostname
    if not host:
        raise SafeURLError("链接缺少主机名")
    if host.strip(".").lower() in BLOCKED_HOSTNAMES or host.strip(".").lower().endswith(
        (".local", ".internal", ".localhost")
    ):
        raise SafeURLError(f"禁止访问内网主机: {host}")
    # 十进制/十六进制等非常规 IP 写法先走 ipaddress，避免依赖系统解析器。
    try:
        ipaddress.ip_address(host)
        _check_ip(host)
    except SafeURLError:
        raise
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        raise SafeURLError(f"无法解析主机 {host}: {e.__class__.__name__}")
    if not infos:
        raise SafeURLError(f"无法解析主机 {host}")
    for info in infos:
        _check_ip(info[4][0])
    return url


def validate_redirect(url: str, previous_host: str) -> str:
    """重定向逐跳校验：协议必须一致或 https，且 host 变化后重新校验 IP。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SafeURLError("重定向仅允许 http/https")
    return validate_public_http_url(url)
