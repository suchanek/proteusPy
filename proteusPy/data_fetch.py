"""
Fetch proteusPy's large data files from the GitHub Release that carries them.

The disulfide database (``PDB_all_ss.pkl``) and the prebuilt loaders
(``PDB_SS_ALL_LOADER.pkl``, ``PDB_SS_SUBSET_LOADER.pkl``) are hundreds of
megabytes, so they are published as assets on a dedicated data release rather
than living in git or in the wheel. See ``ProteusGlobals`` for the tag and the
expected checksums.

A download is streamed to a ``.part`` file, checksummed, and only then moved
into place, so an interrupted transfer never leaves a truncated pickle behind
for the loader to choke on. If the release asset cannot be fetched at all, the
master list falls back to its Google Drive copy.

Author: Eric G. Suchanek, PhD
License: BSD
"""

# pylint: disable=c0103

import hashlib
import os
from pathlib import Path

import requests

from proteusPy.DisulfideExceptions import DisulfideIOException
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
    DATA_DIR,
    DATA_RELEASE_BASE_URL,
    DATA_RELEASE_FALLBACK_URL,
    DATA_RELEASE_SHA256,
    DATA_RELEASE_TAG,
)

_logger = create_logger(__name__)

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__  # type: ignore
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

# Streamed in 1 MB chunks: large enough that the loop is not the bottleneck on a
# 500 MB asset, small enough to keep the progress bar moving.
_CHUNK_SIZE = 1024 * 1024

__all__ = ["data_asset_url", "fetch_data_file", "sha256_file"]


def data_asset_url(fname: str) -> str:
    """
    Return the release-asset download URL for a data file.

    :param fname: Bare filename of the asset, e.g. ``PDB_all_ss.pkl``.
    :type fname: str
    :return: The full download URL on the data release.
    :rtype: str

    Example:
        >>> from proteusPy.data_fetch import data_asset_url
        >>> data_asset_url("PDB_all_ss.pkl").endswith("/PDB_all_ss.pkl")
        True
    """
    return f"{DATA_RELEASE_BASE_URL}/{fname}"


def sha256_file(path: str | Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """
    Compute the sha256 hex digest of a file, reading it in chunks.

    :param path: File to digest.
    :type path: str | Path
    :param chunk_size: Read size in bytes.
    :type chunk_size: int
    :return: Lowercase hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, verbose: bool = False) -> str:
    """
    Stream ``url`` into ``dest``, returning the sha256 of what was written.

    Writes to ``dest`` with a ``.part`` suffix and moves it into place only on a
    clean transfer, so a failure cannot leave a partial file where the loader
    expects a whole one.
    """
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()

    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))

            with (
                open(part, "wb") as f,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    disable=not verbose,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    digest.update(chunk)
                    bar.update(len(chunk))

        written = part.stat().st_size
        if total and written != total:
            raise OSError(f"truncated download: got {written} of {total} bytes")

        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise

    return digest.hexdigest()


def _fetch_from_release(fname: str, dest: Path, verbose: bool = False) -> None:
    """
    Download ``fname`` from the data release into ``dest`` and verify it.

    A checksum mismatch removes the file and raises, rather than leaving a
    corrupt pickle on disk for the next run to load.
    """
    url = data_asset_url(fname)
    if verbose:
        _logger.info("Downloading %s from release %s...", fname, DATA_RELEASE_TAG)

    actual = _download(url, dest, verbose=verbose)
    expected = DATA_RELEASE_SHA256.get(fname, "")

    if not expected:
        _logger.warning(
            "No checksum on record for %s; downloaded content is unverified. "
            "Populate DATA_RELEASE_SHA256 with `make data-checksums`.",
            fname,
        )
        return

    if actual != expected:
        dest.unlink(missing_ok=True)
        raise DisulfideIOException(
            f"Checksum mismatch for {fname}: expected {expected}, got {actual}"
        )

    if verbose:
        _logger.info("Verified %s (sha256 %s...).", fname, actual[:12])


def fetch_data_file(
    fname: str,
    destdir: str | Path = DATA_DIR,
    verbose: bool = False,
    force: bool = False,
    fallback_url: str | None = None,
) -> Path:
    """
    Ensure a large data file is present locally, downloading it if it is not.

    The file is fetched from the proteusPy data release. If that fails and a
    fallback URL is known — either passed in or registered in
    ``DATA_RELEASE_FALLBACK_URL`` — the Google Drive copy is tried next.

    :param fname: Bare filename to fetch, e.g. ``PDB_all_ss.pkl``.
    :type fname: str
    :param destdir: Directory to fetch into; defaults to the package data directory.
    :type destdir: str | Path
    :param verbose: Log progress and show a progress bar.
    :type verbose: bool
    :param force: Re-download even if the file is already present.
    :type force: bool
    :param fallback_url: Overrides the registered fallback for this file.
    :type fallback_url: str | None
    :raises DisulfideIOException: If neither the release asset nor the fallback works.
    :return: Path to the file on disk.
    :rtype: Path
    """
    dest = Path(destdir) / fname

    if dest.exists() and not force:
        if verbose:
            _logger.info("%s already present; skipping download.", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        _fetch_from_release(fname, dest, verbose=verbose)
        return dest
    except Exception as release_error:
        url = fallback_url or DATA_RELEASE_FALLBACK_URL.get(fname)
        if url is None:
            raise DisulfideIOException(
                f"Could not fetch {fname} from release {DATA_RELEASE_TAG} "
                f"and no fallback is registered for it: {release_error}"
            ) from release_error

        _logger.warning(
            "Release asset for %s unavailable (%s); falling back to Drive.",
            fname,
            release_error,
        )

    # Imported here so gdown is only needed when the fallback is actually used.
    import gdown

    try:
        gdown.download(url, str(dest), quiet=not verbose)
    except Exception as drive_error:
        raise DisulfideIOException(
            f"Could not fetch {fname} from the data release or from Drive: {drive_error}"
        ) from drive_error

    if not dest.exists():
        raise DisulfideIOException(
            f"Drive fallback for {fname} reported success but wrote no file."
        )

    return dest


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
