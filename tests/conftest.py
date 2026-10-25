import pytest

from src import db


@pytest.fixture()
def conn(tmp_path):
    with db.get_conn(str(tmp_path / "test.db")) as c:
        yield c
