from __future__ import annotations

from llm_lab.data import ByteTokenizer


def test_byte_tokenizer_roundtrip_and_eot():
    tokenizer = ByteTokenizer()
    text = "hello 世界"
    ids = tokenizer.encode(text, add_eot=True)
    assert ids[-1] == tokenizer.eot_token_id
    assert tokenizer.decode(ids) == text
