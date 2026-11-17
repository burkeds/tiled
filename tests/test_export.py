import csv
import json
from pathlib import Path

import h5py
import numpy
import pandas
import pytest
import xarray

from tiled.adapters.array import ArrayAdapter
from tiled.adapters.dataframe import DataFrameAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.adapters.xarray import DatasetAdapter
from tiled.catalog import in_memory
from tiled.client import Context, from_context
from tiled.client.utils import ClientError
from tiled.queries import Key
from tiled.server.app import build_app

data = numpy.random.random((10, 10))
temp = 15 + 8 * numpy.random.randn(2, 2, 3)
precip = 10 * numpy.random.rand(2, 2, 3)
lon = [[-99.83, -99.32], [-99.79, -99.23]]
lat = [[42.25, 42.21], [42.63, 42.59]]
tree = MapAdapter(
    {
        "A": ArrayAdapter.from_array(numpy.random.random((100, 100))),
        "B": ArrayAdapter.from_array(numpy.random.random((100, 100, 100))),
        "C": DataFrameAdapter.from_pandas(
            pandas.DataFrame(
                {
                    "x": 1 * numpy.random.random(100),
                    "y": 2 * numpy.random.random(100),
                    "z": 3 * numpy.random.random(100),
                }
            ),
            npartitions=3,
        ),
        "empty_table": DataFrameAdapter.from_pandas(
            pandas.DataFrame({"A": []}), npartitions=1
        ),
        "single_row_table": DataFrameAdapter.from_pandas(
            pandas.DataFrame({"A": [1]}), npartitions=1
        ),
        "single_row_many_columns": DataFrameAdapter.from_pandas(
            pandas.DataFrame({f"column_{i}": [i] for i in range(100)}), npartitions=1
        ),
        "structured_data": MapAdapter(
            {
                "pets": ArrayAdapter.from_array(
                    numpy.array(
                        [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
                        dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
                    )
                ),
                "weather": DatasetAdapter.from_dataset(
                    xarray.Dataset(
                        {
                            "temperature": (["x", "y", "time"], temp),
                            "precipitation": (["x", "y", "time"], precip),
                        },
                        coords={
                            "lon": (["x", "y"], lon),
                            "lat": (["x", "y"], lat),
                            "time": numpy.array(
                                [1, 2, 3]
                            ),  # using ints here so HDF5 can export
                        },
                    )
                ),
            }
        ),
    }
)


@pytest.fixture(scope="module")
def client():
    app = build_app(tree)
    with Context.from_app(app) as context:
        client = from_context(context)
        yield client


# We test a little bit of actual file export, using the tmpdir fixture,
# but we mostly export for a buffer in memory because disk access
# can be very cloud on cloud CI VMs.


def has_csv_header(filepath):
    with open(filepath) as csv_f:
        sniffer = csv.Sniffer()
        has_header = sniffer.has_header(csv_f.read(2048))
        csv_f.seek(0)
    return has_header


@pytest.mark.parametrize("filename", ["numbers.csv", "image.png", "image.tiff"])
def test_export_2d_array(client, filename, tmpdir):
    client["A"].export(Path(tmpdir, filename))


@pytest.mark.parametrize("filename", ["numbers.csv", "spreadsheet.xlsx"])
def test_export_table(client, filename, tmpdir):
    client["C"].export(Path(tmpdir, filename))


@pytest.mark.parametrize("filename", ["numbers.csv"])
def test_csv_mimetype_opt_params(client, filename, tmpdir):
    client["C"].export(Path(tmpdir, filename), format="text/csv;header=absent")
    assert not has_csv_header(Path(tmpdir, filename))
    client["C"].export(Path(tmpdir, filename), format="text/csv;header=present")
    assert has_csv_header(Path(tmpdir, filename))


def test_streaming_export(client, buffer):
    "The application/json-seq format is streamed via a generator."
    client["C"].export(buffer, format="application/json-seq")
    # Verify the RFC 7464 record-separator and line-feed framing.
    buffer.seek(0)
    payload = buffer.read()
    assert payload.startswith(b"\x1e")
    records = [record for record in payload.split(b"\x1e") if record]
    assert len(records) == 100
    for record in records:
        assert record.endswith(b"\n")
        json.loads(record)


def test_streaming_export_empty(client, buffer):
    "The application/json-seq format is streamed via a generator."
    client["empty_table"].export(buffer, format="application/json-seq")
    buffer.seek(0)
    assert buffer.read() == b""


@pytest.mark.parametrize(
    "key,expected",
    [
        ("single_row_table", {"A": 1}),
        ("single_row_many_columns", {f"column_{i}": i for i in range(100)}),
    ],
)
def test_json_sequence_single_row_uses_record_separator(client, key, expected):
    """A one-row JSON sequence must not be mistaken for one JSON document."""
    url = client[key].item["links"]["partition"]
    response = client.context.http_client.get(
        url,
        params={"partition": 0, "format": "application/json-seq"},
    )
    response.raise_for_status()

    assert response.headers["content-type"].split(";", 1)[0] == "application/json-seq"
    assert response.content.startswith(b"\x1e")
    assert response.content.endswith(b"\n")
    assert response.content.count(b"\x1e") == 1
    assert json.loads(response.content[1:]) == expected
    with pytest.raises(json.JSONDecodeError):
        json.loads(response.content)


def test_export_weather_data_var(client, tmpdir, buffer):
    client["structured_data"]["weather"]["temperature"].export(
        buffer, slice=(0,), format="text/csv"
    )


def test_export_weather_all(client, buffer):
    client["structured_data"]["weather"].export(buffer, format="application/x-hdf5")


def test_serialization_error_hdf5_metadata(client, buffer):
    tree = MapAdapter(
        {
            "good": MapAdapter({}, metadata={"a": 1}),
            "bad": MapAdapter({}, metadata={"a": {"b": 1}}),
        }
    )
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)
        client["good"].export(buffer, format="application/x-hdf5")
        with pytest.raises(ClientError, match="contains types or structure"):
            client["bad"].export(buffer, format="application/x-hdf5")


def test_path_as_Path_or_string(client, tmpdir):
    client["A"].export(Path(tmpdir, "test_path_as_path.txt"))
    client["A"].export(str(Path(tmpdir, "test_path_as_str.txt")))


def test_formats(client):
    client.formats
    client["A"].formats
    client["C"].formats


filtered_tree = MapAdapter(
    {
        "a": ArrayAdapter.from_array(numpy.arange(3), metadata={"animal": "dog"}),
        "b": ArrayAdapter.from_array(numpy.arange(3), metadata={"animal": "cat"}),
        "c": ArrayAdapter.from_array(numpy.arange(3), metadata={"animal": "dog"}),
    }
)


@pytest.fixture(scope="module")
def filtered_map_client():
    with Context.from_app(build_app(filtered_tree)) as context:
        yield from_context(context)


@pytest.fixture
def filtered_catalog_client(tmpdir):
    catalog = in_memory(writable_storage=str(tmpdir))
    with Context.from_app(build_app(catalog)) as context:
        client = from_context(context)
        for key, value in filtered_tree.items():
            client.write_array(value.read(), metadata=value.metadata(), key=key)
        yield client


@pytest.fixture(params=("filtered_map_client", "filtered_catalog_client"))
def filtered_client(request: pytest.FixtureRequest):
    yield request.getfixturevalue(request.param)


def test_export_search_results(filtered_client, buffer):
    "Exporting a filtered Node writes only the matching subset."
    results = filtered_client.search(Key("animal") == "dog")
    assert list(results) == ["a", "c"]
    results.export(buffer, format="application/x-hdf5")
    with h5py.File(buffer) as file:
        assert set(file.keys()) == {"a", "c"}


def test_export_search_results_with_fields(filtered_client, buffer):
    "Explicit `fields` narrow the search results further (logical AND)."
    results = filtered_client.search(Key("animal") == "dog")
    results.export(buffer, fields=["a"], format="application/x-hdf5")
    with h5py.File(buffer) as file:
        assert set(file.keys()) == {"a"}


def test_export_search_no_matches(filtered_client, buffer):
    "A search matching nothing exports an empty container."
    results = filtered_client.search(Key("animal") == "fish")
    assert len(results) == 0
    results.export(buffer, format="application/x-hdf5")
    with h5py.File(buffer) as file:
        assert set(file.keys()) == set()


def test_export_unfiltered_still_exports_all(filtered_client, buffer):
    "Without a search, export is unchanged: every child is written."
    filtered_client.export(buffer, format="application/x-hdf5")
    with h5py.File(buffer) as file:
        assert set(file.keys()) == {"a", "b", "c"}
