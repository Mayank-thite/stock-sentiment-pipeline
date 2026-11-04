import pytest


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from pipeline.config import spark_session

    base = tmp_path_factory.mktemp("spark")
    s = spark_session("tests", metastore=base / "metastore")
    yield s
    s.stop()
