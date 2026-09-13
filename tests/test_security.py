# -*- coding: utf-8 -*-
import _isolate  # noqa: F401  必须先于 app.* 导入

import socket
import unittest

from app import security
from app.security import SafeURLError, validate_public_http_url, validate_redirect


class _FakeResolver:
    """把任意域名解析到指定 IP，用来模拟 DNS rebinding。"""

    def __init__(self, ip):
        self.ip = ip
        self.original = socket.getaddrinfo

    def __enter__(self):
        socket.getaddrinfo = lambda host, port, *a, **k: [(2, 1, 6, "", (self.ip, 0))]
        return self

    def __exit__(self, *exc):
        socket.getaddrinfo = self.original
        return False


class SchemeTest(unittest.TestCase):
    def test_rejects_non_http_schemes(self):
        for url in ["file:///etc/passwd", "ftp://host/a.csv", "gopher://x/", "javascript:alert(1)",
                    "FILE://c:/windows/win.ini"]:
            with self.subTest(url=url):
                with self.assertRaises(SafeURLError):
                    validate_public_http_url(url)

    def test_rejects_missing_host(self):
        for url in ["http://", "not a url at all", "", "   ", "http:///relative/path"]:
            with self.subTest(url=url):
                with self.assertRaises(SafeURLError):
                    validate_public_http_url(url)


class BlockedHostnameTest(unittest.TestCase):
    def test_blocked_names_do_not_need_dns(self):
        # 这些必须在解析之前就拒掉，否则会被 DNS 结果绕过
        for url in ["http://localhost:8765/api/runs", "http://LOCALHOST./x",
                    "http://metadata.google.internal/computeMetadata/v1/",
                    "http://metadata.goog/x", "http://nas.internal/share",
                    "http://router.local/x", "http://evil.localhost/x"]:
            with self.subTest(url=url):
                with self.assertRaises(SafeURLError):
                    validate_public_http_url(url)


class IpLiteralTest(unittest.TestCase):
    def test_rejects_internal_ranges(self):
        bad = [
            "http://127.0.0.1/x", "http://127.0.0.53/x", "http://127.1/x",
            "http://[::1]/x", "http://0.0.0.0/x",
            "http://10.0.0.5/x", "http://172.16.0.1/x", "http://172.31.255.255/x",
            "http://192.168.1.1/x", "http://169.254.169.254/latest/meta-data/",
            "http://224.0.0.1/x", "http://255.255.255.255/x",
            "http://[fe80::1]/x", "http://[fc00::1]/x",
        ]
        for url in bad:
            with self.subTest(url=url):
                with self.assertRaises(SafeURLError):
                    validate_public_http_url(url)

    def test_decimal_packed_loopback_is_still_caught(self):
        # 127.0.0.1 的十进制写法 2130706433 会被 ipaddress 拒为非法字面量，
        # 而 urlparse 也拿不到合法 hostname；两种结果都必须是"拒绝"。
        with self.assertRaises(SafeURLError):
            validate_public_http_url("http://2130706433/x")


class RebindingTest(unittest.TestCase):
    def test_public_name_resolving_to_loopback_is_rejected(self):
        with _FakeResolver("127.0.0.1"):
            with self.assertRaises(SafeURLError):
                validate_public_http_url("http://evil.example.com/a.csv")

    def test_public_name_resolving_to_cloud_metadata_is_rejected(self):
        with _FakeResolver("169.254.169.254"):
            with self.assertRaises(SafeURLError):
                validate_public_http_url("https://attacker.example/a.zip")

    def test_public_name_resolving_to_public_ip_is_allowed(self):
        with _FakeResolver("93.184.216.34"):
            self.assertEqual(validate_public_http_url("https://example.com/a.csv"),
                             "https://example.com/a.csv")

    def test_unresolvable_host_is_rejected(self):
        def boom(*a, **k):
            raise socket.gaierror("no such host")

        original = socket.getaddrinfo
        socket.getaddrinfo = boom
        try:
            with self.assertRaises(SafeURLError):
                validate_public_http_url("https://does-not-exist.invalid/a.csv")
        finally:
            socket.getaddrinfo = original


class RedirectTest(unittest.TestCase):
    def test_redirect_to_internal_address_is_rejected(self):
        with self.assertRaises(SafeURLError):
            validate_redirect("http://169.254.169.254/latest/meta-data", "example.com")

    def test_redirect_to_non_http_is_rejected(self):
        with self.assertRaises(SafeURLError):
            validate_redirect("file:///etc/passwd", "example.com")


class CheckIpTest(unittest.TestCase):
    def test_non_ip_string_raises(self):
        with self.assertRaises(SafeURLError):
            security._check_ip("not-an-ip")


if __name__ == "__main__":
    unittest.main()
