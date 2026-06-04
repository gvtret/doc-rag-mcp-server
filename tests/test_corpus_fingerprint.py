from doc_rag.raglib.pipeline import compute_corpus_fingerprint


def test_fingerprint_sort_invariant():
    a = compute_corpus_fingerprint([{"sha256": "bb"}, {"sha256": "aa"}])
    b = compute_corpus_fingerprint([{"sha256": "aa"}, {"sha256": "bb"}])
    assert a == b


def test_fingerprint_empty():
    assert (
        compute_corpus_fingerprint([])
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
