from sre_shared.config import load_corpus_config, load_limits, load_retrieval_defaults


def test_limits_load() -> None:
    limits = load_limits()
    groq = limits.model_limits("groq", "openai/gpt-oss-20b")
    assert groq.tpd == 200_000
    assert limits.model_limits("groq", "nope").tpd is None


def test_retrieval_defaults_load() -> None:
    d = load_retrieval_defaults()
    assert d.rrf_k == 60
    assert d.hnsw.ef_search >= 1


def test_corpus_config_load() -> None:
    c = load_corpus_config()
    assert c.license == "CC-BY-4.0"
