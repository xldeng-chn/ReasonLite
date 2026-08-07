"""Tests for the ReasonLite SFT entry (dataset normalization + tokenizer hashing).

Run with: python3 -m pytest tests/test_sft_reasonlite.py
Or standalone: python3 tests/test_sft_reasonlite.py

Pure-stdlib: sft_reasonlite.py keeps its open-r1/torch/datasets imports inside
`if __name__ == "__main__"`, so the module-level helpers import and test here
without the training stack installed.
"""

import importlib.util
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_ENTRY = os.path.join(REPO_ROOT, "train", "sft_reasonlite.py")


def _load_sft_module():
    spec = importlib.util.spec_from_file_location("sft_reasonlite", SFT_ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sft = _load_sft_module()


class FakeAddedToken:
    """Stand-in for transformers' AddedToken (duck-typed attribute surface)."""

    def __init__(self, content, special=False, lstrip=False, rstrip=False,
                 single_word=False, normalized=False):
        self.content = content
        self.special = special
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.single_word = single_word
        self.normalized = normalized


def _fake_tokenizer(**overrides):
    """Build a duck-typed tokenizer carrying every attribute the identity reads.

    Mirrors the Qwen2TokenizerFast surface seen in the parity runs; each test
    overrides exactly one field to prove that field participates in the key.
    """
    attrs = {
        "name_or_path": "/user/dengxianglong/models/Qwen3-0.6B",
        "vocab_size": 151643,
        "model_max_length": 131072,
        "truncation_side": "right",
        "padding_side": "right",
        "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
        "special_tokens_map": {
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
        },
        "added_tokens_decoder": {
            151643: FakeAddedToken("<|endoftext|>", special=True),
            151667: FakeAddedToken("<think>"),
            151668: FakeAddedToken("</think>"),
        },
    }
    attrs.update(overrides)
    return type("FakeTokenizer", (), attrs)()


class TestTokenizerIdentity(unittest.TestCase):
    """The identity is the cache key that replaces datasets' random fallback.

    Two properties matter and they pull in opposite directions:
      - stable: equivalent tokenizers must produce the same key, or the cache
        never hits and the 12-minute tokenize is paid on every launch;
      - complete: anything that changes tokenize OUTPUT must change the key, or
        a stale cache is silently reused and training eats the wrong data.
    Each test below pins one side or the other.
    """

    def test_identity_is_stable_across_calls(self):
        tok = _fake_tokenizer()
        self.assertEqual(sft.tokenizer_identity(tok), sft.tokenizer_identity(tok))

    def test_equivalent_tokenizers_share_identity(self):
        # Independently constructed but equivalent -> same key, or reruns never
        # hit the cache (the bug this whole change exists to fix).
        self.assertEqual(
            sft.tokenizer_identity(_fake_tokenizer()),
            sft.tokenizer_identity(_fake_tokenizer()),
        )

    def test_identity_carries_no_memory_address(self):
        # A default repr() leaks 0x... addresses, which differ per process and
        # would reintroduce the random-fingerprint behaviour we are removing.
        self.assertNotIn("0x", sft.tokenizer_identity(_fake_tokenizer()))

    def test_added_token_order_does_not_change_identity(self):
        # dict iteration order must not leak into the key.
        forward = _fake_tokenizer(added_tokens_decoder={
            151643: FakeAddedToken("<|endoftext|>", special=True),
            151667: FakeAddedToken("<think>"),
        })
        reverse = _fake_tokenizer(added_tokens_decoder={
            151667: FakeAddedToken("<think>"),
            151643: FakeAddedToken("<|endoftext|>", special=True),
        })
        self.assertEqual(
            sft.tokenizer_identity(forward), sft.tokenizer_identity(reverse)
        )


class TestTokenizerIdentityCompleteness(unittest.TestCase):
    """Every attribute that can change tokenize output must change the key."""

    def _assert_invalidates(self, **overrides):
        base = sft.tokenizer_identity(_fake_tokenizer())
        changed = sft.tokenizer_identity(_fake_tokenizer(**overrides))
        self.assertNotEqual(base, changed, f"identity ignored {list(overrides)}")

    def test_chat_template_change_invalidates(self):
        # The template renders messages into the text that gets tokenized, so it
        # decides the output outright -- and it lives in neither vocab_size nor
        # name_or_path. This is the attribute most likely to be missed.
        self._assert_invalidates(chat_template="{{ 'totally different' }}")

    def test_chat_template_absent_vs_present_invalidates(self):
        self._assert_invalidates(chat_template=None)

    def test_name_or_path_change_invalidates(self):
        self._assert_invalidates(name_or_path="/user/dengxianglong/models/Qwen3-8B")

    def test_vocab_size_change_invalidates(self):
        self._assert_invalidates(vocab_size=32000)

    def test_added_tokens_change_invalidates(self):
        self._assert_invalidates(added_tokens_decoder={
            151643: FakeAddedToken("<|endoftext|>", special=True),
            151667: FakeAddedToken("<think>"),
            151669: FakeAddedToken("<new_special>", special=True),
        })

    def test_added_token_flag_change_invalidates(self):
        # Same content and id, different lstrip -> different tokenization.
        self._assert_invalidates(added_tokens_decoder={
            151667: FakeAddedToken("<think>", lstrip=True),
        })

    def test_special_tokens_map_change_invalidates(self):
        self._assert_invalidates(special_tokens_map={
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
        })

    def test_model_max_length_change_invalidates(self):
        # Drives truncation, which is a separate map step in the same chain.
        self._assert_invalidates(model_max_length=32768)

    def test_truncation_side_change_invalidates(self):
        self._assert_invalidates(truncation_side="left")

    def test_padding_side_change_invalidates(self):
        self._assert_invalidates(padding_side="left")

    def test_tokenizer_class_change_invalidates(self):
        # Two classes can share every attribute above and still tokenize
        # differently (slow vs fast backend), so the class name is part of it.
        tok = _fake_tokenizer()
        other = type("DifferentTokenizer", (), dict(vars(type(tok))))()
        self.assertNotEqual(
            sft.tokenizer_identity(tok), sft.tokenizer_identity(other)
        )

    def test_version_tag_invalidates(self):
        # Manual escape hatch: bump the tag to void every cached key on purpose.
        tok = _fake_tokenizer()
        self.assertNotEqual(
            sft.tokenizer_identity(tok),
            sft.tokenizer_identity(tok, version="v2"),
        )


class TestRegisterStableTokenizerHash(unittest.TestCase):
    """The registration seam. datasets is absent locally, so this covers the
    contract that does not need it; the reducer itself is exercised on-cluster.
    """

    def test_register_is_exposed(self):
        self.assertTrue(callable(sft.register_stable_tokenizer_hash))

    def test_register_fails_fast_without_datasets(self):
        # No silent no-op: if the pickler seam ever moves, the run must stop
        # rather than quietly fall back to recomputing every launch.
        if importlib.util.find_spec("datasets") is not None:
            self.skipTest("datasets installed; import-error path not reachable")
        with self.assertRaises(ImportError):
            sft.register_stable_tokenizer_hash(_fake_tokenizer())


if __name__ == "__main__":
    unittest.main()
