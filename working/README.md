# Blissdata → Tiled Adapter

A [Tiled](https://github.com/bluesky/tiled) adapter that exposes a
[Blissdata](https://gitlab.esrf.fr/bliss/blissdata) Redis database as a
browsable, sliceable data service.

## Tree structure served

```
DataStoreAdapter          ← root MapAdapter (one per Redis instance)
└── esrf__scan__<ULID>    ← ScanAdapter per acquisition
    ├── metadata           ← TableAdapter  (1-row, all identity + info fields)
    ├── timer__epoch       ← ArrayAdapter  (numeric stream → numpy array)
    ├── diode__diode       ← ArrayAdapter  (array stream  → numpy array)
    └── config             ← TableAdapter  (json stream   → pandas DataFrame)
```

Colons (`:`) in Blissdata keys are replaced with double-underscores (`__`)
to produce valid Tiled node keys.

## Installation

```bash
pip install "tiled[server]" blissdata numpy pandas
```

## Usage

### As a Python object

```python
from blissdata_adapter import DataStoreAdapter

adapter = DataStoreAdapter.from_uri("redis://localhost:6379")

# List scans
print(list(adapter.keys()))

# Navigate into a scan
scan = adapter["esrf__scan__01HCYP7ARJ4NSG9VGPGDX6SN2X"]

# Read scan metadata
meta_df = scan["metadata"].read()
print(meta_df)

# Read a numeric stream as a numpy array
data = scan["timer__epoch"].read()   # → numpy array shape (N,)

# Read an array stream (e.g. 2-D detector)
images = scan["detector__image"].read()   # → numpy array shape (N, H, W)

# Read a JSON stream as a DataFrame
df = scan["config"].read()
```

### As a Tiled server

```bash
tiled serve config config.yml
```

Then connect from anywhere:

```python
from tiled.client import from_uri

c = from_uri("http://localhost:8000?api_key=<printed_at_startup>")

# List scans
print(list(c.keys()))

# Slice a large detector stream – only transfers the requested chunk
images = c["esrf__scan__01HCYP7ARJ4NSG9VGPGDX6SN2X"]["detector__image"][0:10]

# Search by metadata key (Tiled built-in search)
from tiled.queries import Key
results = c.search(Key("state") == "ScanState.CLOSED")
```

## Filtering scans at startup

Pass `search_kwargs` to limit which scans are loaded:

```python
adapter = DataStoreAdapter(
    redis_url="redis://localhost:6379",
    search_kwargs={"session": "optics_lab", "proposal": "hc12345"},
)
```

Or in `config.yml`:

```yaml
args:
  redis_url: "redis://localhost:6379"
  search_kwargs:
    session: "optics_lab"
    proposal: "hc12345"
```

## Picking up new scans at runtime

```python
# After new scans have been acquired:
adapter.refresh()
```

For a long-running server you can refresh on a schedule:

```python
import threading, time

def auto_refresh(adapter, interval=30):
    while True:
        time.sleep(interval)
        try:
            adapter.refresh()
        except Exception as exc:
            print("refresh failed:", exc)

threading.Thread(target=auto_refresh, args=(adapter,), daemon=True).start()
```

## Design notes

| Blissdata concept | Tiled node type | Notes |
|---|---|---|
| `DataStore` | `MapAdapter` (root) | Connects to Redis on construction |
| `Scan` | `MapAdapter` (`ScanAdapter`) | Identity + info fields as metadata |
| Numeric/array `Stream` | `ArrayAdapter` | dtype and shape preserved; shape is `(N, *point_shape)` |
| JSON `Stream` | `TableAdapter` | Columns are the JSON keys of the first point |
| Scan metadata | `TableAdapter` | 1-row table with all scalar identity and info fields |

### Caveats

- **Live streams**: data is snapshotted at node-creation time. Call
  `adapter.refresh()` (or re-create the `ScanAdapter`) to see new points.
  For true live streaming, wire the Cursor API into a background thread and
  update the underlying array in place.
- **Memory eviction**: if Redis has already evicted old stream entries, early
  indices will raise `IndexNoMoreThereError`; the adapter catches this and
  returns whatever data is still available.
- **Identity model**: the adapter probes common ESRF identity fields
  (`name`, `number`, `session`, `proposal`, `collection`, `dataset`, `path`)
  but gracefully skips any that are absent.
