from sre_shared.models import RetrievalRequest
from sre_shared.stamps import config_hash


def test_config_hash_is_stable_and_order_free() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert config_hash(RetrievalRequest(query="q")) == config_hash(RetrievalRequest(query="q"))
