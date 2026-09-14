#!/usr/bin/env python3
# test_data_fetch.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-09-14 -egs-

"""
Unit tests for proteusPy.data_fetch.

The large .pkl files are release assets, so these tests stand a throwaway HTTP
server in for the release and point DATA_RELEASE_BASE_URL at it. Nothing here
touches the network or the real data directory.
"""

# pylint: disable=C0115,C0116,C0103

import functools
import hashlib
import http.server
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path

from proteusPy import data_fetch
from proteusPy.data_fetch import data_asset_url, fetch_data_file, sha256_file
from proteusPy.DisulfideExceptions import DisulfideIOException

ASSET = "TEST_ASSET.pkl"

# Larger than the 1 MB download chunk, so the streaming loop really iterates.
PAYLOAD = b"proteusPy" * (350 * 1024)
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class DataFetchTestCase(unittest.TestCase):
    """Serve ASSET from a local directory and fetch it into another."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.served = root / "served"
        cls.served.mkdir()
        (cls.served / ASSET).write_bytes(PAYLOAD)

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(cls.served)
        )
        socketserver.TCPServer.allow_reuse_address = True
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls._tmp.cleanup()

    def setUp(self):
        self._saved = (
            data_fetch.DATA_RELEASE_BASE_URL,
            data_fetch.DATA_RELEASE_SHA256,
            data_fetch.DATA_RELEASE_FALLBACK_URL,
        )
        data_fetch.DATA_RELEASE_BASE_URL = self.base_url
        data_fetch.DATA_RELEASE_SHA256 = {ASSET: DIGEST}
        # No Drive fallback: a failure here should surface, not be papered over.
        data_fetch.DATA_RELEASE_FALLBACK_URL = {}

        self._dest_tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self._dest_tmp.name)

    def tearDown(self):
        (
            data_fetch.DATA_RELEASE_BASE_URL,
            data_fetch.DATA_RELEASE_SHA256,
            data_fetch.DATA_RELEASE_FALLBACK_URL,
        ) = self._saved
        self._dest_tmp.cleanup()

    def test_asset_url_is_tag_relative(self):
        self.assertEqual(data_asset_url(ASSET), f"{self.base_url}/{ASSET}")

    def test_download_and_verify(self):
        path = fetch_data_file(ASSET, destdir=self.dest)
        self.assertEqual(path.read_bytes(), PAYLOAD)
        self.assertEqual(sha256_file(path), DIGEST)

    def test_existing_file_is_not_refetched(self):
        path = fetch_data_file(ASSET, destdir=self.dest)
        path.write_bytes(b"sentinel")
        self.assertEqual(fetch_data_file(ASSET, destdir=self.dest).read_bytes(), b"sentinel")

    def test_force_refetches(self):
        path = fetch_data_file(ASSET, destdir=self.dest)
        path.write_bytes(b"sentinel")
        self.assertEqual(
            fetch_data_file(ASSET, destdir=self.dest, force=True).read_bytes(), PAYLOAD
        )

    def test_checksum_mismatch_raises_and_removes_the_file(self):
        data_fetch.DATA_RELEASE_SHA256 = {ASSET: "0" * 64}
        with self.assertRaises(DisulfideIOException) as ctx:
            fetch_data_file(ASSET, destdir=self.dest)
        self.assertIn("Checksum mismatch", str(ctx.exception))
        self.assertFalse((self.dest / ASSET).exists())
        self.assertEqual(list(self.dest.glob("*.part")), [])

    def test_missing_asset_without_fallback_raises(self):
        with self.assertRaises(DisulfideIOException) as ctx:
            fetch_data_file("NO_SUCH_ASSET.pkl", destdir=self.dest)
        self.assertIn("no fallback is registered", str(ctx.exception))
        self.assertEqual(list(self.dest.glob("*.part")), [])

    def test_unchecksummed_asset_still_downloads(self):
        data_fetch.DATA_RELEASE_SHA256 = {}
        self.assertEqual(fetch_data_file(ASSET, destdir=self.dest).read_bytes(), PAYLOAD)

    def test_destination_directory_is_created(self):
        nested = self.dest / "a" / "b"
        self.assertEqual(fetch_data_file(ASSET, destdir=nested).read_bytes(), PAYLOAD)


if __name__ == "__main__":
    unittest.main()
