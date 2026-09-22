from romanian_news.identity import canonical_json, sha256


def test_canonical_json_and_sha256_preserve_identity_bytes() -> None:
    content = canonical_json({"z": "ș", "a": [1, True, None], "floats": [1.25, 1e20, -0.0]})

    assert content == b'{"a":[1,true,null],"floats":[1.25,1e+20,-0.0],"z":"\xc8\x99"}'
    assert sha256(content) == "656944304226321b1b801046a010c12819de056ffd3e13bbfbb39da5bf4f39df"
