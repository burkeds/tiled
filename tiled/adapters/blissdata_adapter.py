"""
Tiled Adapter for Blissdata Redis databases
============================================

Exposes a Blissdata DataStore as a Tiled tree with the following structure:

    DataStoreAdapter  (root container)
    └── <scan_key>  ScanAdapter  (one per scan, keyed by ULID)
        ├── metadata  (scan identity + info, served as a 1-row table)
        └── <stream_name>  StreamAdapter  (one per stream)
                           → numeric/array streams  → ArrayAdapter
                           → json streams           → TableAdapter

Usage – Python API
------------------
    from blissdata_adapter import DataStoreAdapter

    adapter = DataStoreAdapter.from_uri("redis://localhost:6379")

Usage – Tiled config (config.yml)
-----------------------------------
    trees:
      - tree: blissdata_adapter:DataStoreAdapter
        path: /
        args:
          redis_url: "redis://localhost:6379"

    # To restrict to a specific session or proposal use searchable_kwargs:
      - tree: blissdata_adapter:DataStoreAdapter
        path: /
        args:
          redis_url: "redis://localhost:6379"
          search_kwargs:
            session: "my_session"

Requirements
------------
    pip install tiled[server] blissdata numpy pandas
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from tiled.adapters.array import ArrayAdapter, ArrayStructure
from tiled.adapters.mapping import MapAdapter
from tiled.adapters.table import TableAdapter
from tiled.structures.core import Spec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_scan_key(key: str) -> str:
    """Convert a Blissdata scan key to a safe Tiled node key.

    Blissdata keys look like ``esrf:scan:01HCYP7ARJ4NSG9VGPGDX6SN2X``.
    Tiled node keys must not contain ``/`` or start with ``.``.
    We replace ``:`` with ``__`` to keep the key readable.
    """
    return key.replace(":", "__")


def _scan_key_from_tiled_key(tiled_key: str) -> str:
    """Reverse of _safe_scan_key."""
    return tiled_key.replace("__", ":")


# ---------------------------------------------------------------------------
# StreamAdapter
# ---------------------------------------------------------------------------

class StreamAdapter:
    """Adapter for a single Blissdata stream.

    Numeric/array streams are exposed as an ``ArrayAdapter``; JSON streams are
    exposed as a ``TableAdapter`` (one row per point, columns are the JSON keys
    discovered from the first point).

    Because streams can still be live (STARTED state), data is read lazily on
    every ``read()`` call so that newly arrived points are always visible.
    """

    def __new__(cls, stream, scan_metadata: dict) -> Any:  # type: ignore[return]
        """Factory: return the right Tiled adapter for this stream type."""
        return cls._build(stream, scan_metadata)

    @staticmethod
    def _build(stream, scan_metadata: dict) -> Any:
        dtype_str = stream.info.get("dtype", "float64")

        if dtype_str == "json":
            return _JsonStreamAdapter(stream, scan_metadata)
        else:
            return _NumericStreamAdapter(stream, scan_metadata)


class _NumericStreamAdapter(ArrayAdapter):
    """ArrayAdapter backed by a Blissdata numeric/array stream."""

    def __init__(self, stream, scan_metadata: dict) -> None:
        self._stream = stream

        # Determine dtype and per-point shape from stream info
        dtype = np.dtype(stream.info.get("dtype", "float64"))
        point_shape = tuple(stream.info.get("shape", []))

        # We don't know the final length upfront for live streams.
        # We read all available data immediately; for production use
        # you could add a dask.array.from_delayed for lazy loading.
        try:
            data = np.array(stream[:], dtype=dtype)
        except Exception:
            # Stream may be empty or sealed with 0 points
            data = np.empty((0,) + point_shape, dtype=dtype)

        # If the stream has only scalar points the data shape is (N,);
        # if it has array points the shape is (N, *point_shape).
        if data.ndim == 0:
            data = data.reshape(1)

        metadata = {
            "stream_name": stream.name,
            "dtype": str(dtype),
            "point_shape": list(point_shape),
            "scan": scan_metadata,
        }

        structure = ArrayStructure(
            shape=data.shape,
            data_type=dtype,
            chunks = ((3,), (3,))
        )

        super().__init__(
            data,
            metadata=metadata,
            specs=[Spec("BlissdataStream")],
            structure=structure
        )

    def refresh(self) -> "_NumericStreamAdapter":
        """Return a fresh adapter with up-to-date data (for live streams)."""
        return _NumericStreamAdapter(self._stream, self.metadata.get("scan", {}))


class _JsonStreamAdapter(TableAdapter):
    """TableAdapter backed by a Blissdata JSON stream."""

    def __init__(self, stream, scan_metadata: dict) -> None:
        self._stream = stream

        try:
            raw_points = stream[:]
        except Exception:
            raw_points = []

        if raw_points:
            df = pd.DataFrame(raw_points)
        else:
            df = pd.DataFrame()

        metadata = {
            "stream_name": stream.name,
            "dtype": "json",
            "scan": scan_metadata,
        }

        super().__init__(
            TableAdapter.from_pandas(
                df,
                metadata=metadata,
                specs=[Spec("BlissdataJsonStream")],
            )._partitions,
            structure=TableAdapter.from_pandas(df).structure(),
            metadata=metadata,
            specs=[Spec("BlissdataJsonStream")],
        )


# ---------------------------------------------------------------------------
# ScanAdapter
# ---------------------------------------------------------------------------

class ScanAdapter(MapAdapter):
    """A container adapter for one Blissdata scan.

    Children:
      - ``metadata``  — 1-row ``TableAdapter`` with all scalar fields
      - one child per stream, named by ``stream.name``
    """

    def __init__(self, scan) -> None:
        scan.update(block=False)  # refresh to latest state without blocking

        scan_meta = self._extract_metadata(scan)

        children: Dict[str, Any] = {}

        # ---- metadata table ------------------------------------------------
        meta_df = pd.DataFrame([scan_meta])
        children["metadata"] = TableAdapter.from_pandas(
            meta_df,
            metadata={"description": "Scan identity and info fields"},
            specs=[Spec("BlissdataScanMetadata")],
        )

        # ---- streams -------------------------------------------------------
        for stream_name, stream in scan.streams.items():
            safe_name = stream_name.replace(":", "__").replace("/", "__")
            try:
                children[safe_name] = StreamAdapter._build(stream, scan_meta)
            except Exception as exc:
                logger.warning(
                    "Could not build adapter for stream %r of scan %r: %s",
                    stream_name,
                    scan.key,
                    exc,
                )

        super().__init__(
            children,
            metadata=scan_meta,
            specs=[Spec("BlissdataScan")],
        )

    @staticmethod
    def _extract_metadata(scan) -> dict:
        """Flatten the scan identity + info into a plain dict."""
        meta: dict = {}

        # Identity fields (vary by model, e.g. ESRFIdentityModel)
        for field in ("name", "number", "data_policy", "proposal",
                      "collection", "dataset", "path", "session"):
            try:
                meta[field] = getattr(scan, field)
            except AttributeError:
                pass

        meta["key"] = scan.key
        meta["state"] = str(scan.state)

        # Free-form info dict — flatten one level
        try:
            for k, v in (scan.info or {}).items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    meta[f"info.{k}"] = v
                else:
                    meta[f"info.{k}"] = str(v)
        except Exception:
            pass

        return meta


# ---------------------------------------------------------------------------
# DataStoreAdapter  (root)
# ---------------------------------------------------------------------------

class DataStoreAdapter(MapAdapter):
    """Root Tiled adapter for an entire Blissdata DataStore.

    Each scan in the DataStore is exposed as a child ``ScanAdapter``, keyed by
    a URL-safe version of the scan's Redis key.

    Parameters
    ----------
    redis_url:
        Redis connection URL, e.g. ``redis://localhost:6379``.
    search_kwargs:
        Optional keyword arguments forwarded to
        ``DataStore.search_existing_scans()``.  Can be used to filter by
        ``session``, ``proposal``, ``dataset``, etc.
    metadata:
        Optional metadata dict attached to the root node.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        search_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        from blissdata import DataStore  # imported here to keep import optional

        self._ds = DataStore(redis_url)
        self._search_kwargs = search_kwargs or {}

        scan_map = self._build_scan_map()

        super().__init__(
            scan_map,
            metadata=metadata or {"redis_url": redis_url},
            specs=[Spec("BlissdataDataStore")],
        )

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_uri(
        cls,
        redis_url: str,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "DataStoreAdapter":
        """Construct from a Redis URL string."""
        return cls(redis_url=redis_url, search_kwargs=search_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scan_map(self) -> Dict[str, Any]:
        """Load all matching scans and build the child map lazily."""
        from blissdata.exceptions import NoScanAvailable

        children: Dict[str, Any] = {}

        try:
            _timestamp, keys = self._ds.search_existing_scans(**self._search_kwargs)
        except NoScanAvailable:
            logger.info("No scans found in DataStore.")
            return children
        except TypeError:
            # Some versions return (timestamp, keys), others just keys
            keys = []

        for key in keys:
            tiled_key = _safe_scan_key(key)
            # Use a closure so each key is captured correctly
            children[tiled_key] = self._make_scan_adapter(key)

        return children

    def _make_scan_adapter(self, scan_key: str) -> ScanAdapter:
        """Load a scan from Redis and return a ScanAdapter."""
        scan = self._ds.load_scan(scan_key)
        return ScanAdapter(scan)

    # ------------------------------------------------------------------
    # Refresh support – call this to pick up new scans without restarting
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the scan list from Redis.

        Useful when new scans have been added since the adapter was created.
        Call this manually or wire it up to a background thread.
        """
        new_map = self._build_scan_map()
        # Update the internal mapping in-place
        self._mapping.clear()
        self._mapping.update(new_map)
