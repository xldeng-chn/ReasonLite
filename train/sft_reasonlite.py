# ReasonLite SFT entry: run open-r1's SFT with a parquet -> messages adapter.
#
# The published ReasonLite-Dataset (parquet) exposes columns
#   [prompt, answer, expected_answer, vote, problem_source]
# with no `messages`/`completion`/`text` column. TRL 0.18's SFTTrainer sees a
# bare `prompt` column, assumes prompt-completion format, and looks for a
# `completion` column -> KeyError: 'completion'.
#
# open-r1's sft.py hands `get_dataset(...)[split]` straight to SFTTrainer with
# no format normalization (it assumes datasets already arrive as `messages`,
# like its default Mixture-of-Thoughts). ReasonLite's does not.
#
# This entry intercepts open-r1's public `get_dataset` seam: after it loads the
# DatasetDict, we map each row to the conversational `messages` shape, then
# call open-r1's own `main` (training loop is reused, not copied). No open-r1
# source is edited, and the dataset is not materialized to disk.
#
# The `messages` role structure (user=prompt, assistant=answer) matches the
# ReasonLite SFT sample schema defined by utils/saving_to_training_format.py.

import json

# Metadata columns dropped after building `messages` (SFTTrainer only needs
# the conversation; the rest are dataset provenance).
_DROP_COLUMNS = ["prompt", "answer", "expected_answer", "vote", "problem_source"]

# Bump to deliberately void every cached tokenize result (see tokenizer_identity).
_TOKENIZER_IDENTITY_VERSION = "v1"

# AddedToken fields that change how a token is matched during tokenization.
_ADDED_TOKEN_FIELDS = ("content", "special", "lstrip", "rstrip",
                       "single_word", "normalized")


def tokenizer_identity(tokenizer, version=_TOKENIZER_IDENTITY_VERSION):
    """Build a stable, dill-hashable identity string for a tokenizer.

    Pure (attribute reads only) so it unit-tests against a duck-typed fake.

    WHY THIS EXISTS. `datasets` keys each `map` cache entry on
    hash(input fingerprint + transform + fn_kwargs). SFTTrainer passes the
    tokenizer itself in fn_kwargs, and `datasets`' own reducer for
    PreTrainedTokenizerBase pickles the instance `__dict__` -- which on a fast
    tokenizer holds the Rust-backed `tokenizers.Tokenizer`. That pickle fails,
    `update_fingerprint` swallows the failure in a bare except and falls back to
    a RANDOM fingerprint. The tokenize result is still written to GPFS, but
    under a key no later run can compute: every launch re-tokenizes 4.33M rows
    (~12 min measured on the 2000-step parity runs) and leaves 64 orphan arrow
    shards behind. Reducing the tokenizer to this identity instead makes the key
    content-derived, so the cache is actually reusable.

    COMPLETENESS IS THE SAFETY PROPERTY. Declaring two tokenizers equal means
    declaring their tokenize output equal. Anything omitted here that affects
    output becomes a silent stale-cache hit -- training would consume data from
    a different configuration with no error. Hence chat_template is included
    (it renders messages into the text that gets tokenized, and appears in
    neither vocab_size nor name_or_path), as are the added-token match flags and
    the truncation/padding sides. When in doubt, bump the version tag above.
    """
    added = sorted(
        [int(tid)] + [_stringify(getattr(tok, f, None)) for f in _ADDED_TOKEN_FIELDS]
        for tid, tok in dict(getattr(tokenizer, "added_tokens_decoder", {})).items()
    )
    payload = {
        "version": version,
        # Two classes can carry identical attributes and still tokenize
        # differently (slow vs fast backend), so the class name is part of it.
        "class": type(tokenizer).__name__,
        "name_or_path": _readable(tokenizer, "name_or_path"),
        "vocab_size": _readable(tokenizer, "vocab_size"),
        "model_max_length": _readable(tokenizer, "model_max_length"),
        "truncation_side": _readable(tokenizer, "truncation_side"),
        "padding_side": _readable(tokenizer, "padding_side"),
        "chat_template": _readable(tokenizer, "chat_template"),
        "special_tokens_map": _readable(tokenizer, "special_tokens_map"),
        "added_tokens": added,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _readable(obj, attr):
    """Read one attribute as a JSON-stable scalar/container.

    Normalising here keeps default reprs out of the payload: a repr embeds a
    0x memory address, which differs per process and would reintroduce exactly
    the per-run instability this function exists to remove.
    """
    value = getattr(obj, attr, None)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return sorted(str(_stringify(v)) for v in value)
    return _stringify(value)


def _stringify(value):
    """Reduce one value to a JSON-stable scalar.

    Objects exposing `content` (AddedToken) carry their text there; anything
    else falls back to str(). Guarded by test_identity_carries_no_memory_address.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(getattr(value, "content", value))


def register_stable_tokenizer_hash(tokenizer):
    """Make `datasets` fingerprint this tokenizer by identity, not by pickle.

    Registers a reducer for the tokenizer's concrete class in the `datasets`
    pickler dispatch. `Pickler.save` only lazy-registers its own
    PreTrainedTokenizerBase handler when the type is absent from the dispatch,
    so registering the concrete class up front takes precedence over it.

    Raises ImportError if the pickler seam has moved (both `pklregister` and the
    dispatch table are private API). Failing loudly is the point: a silent no-op
    would look identical to success while every launch kept re-tokenizing.
    """
    from datasets.utils._dill import pklregister

    def _save_tokenizer_by_identity(pickler, obj):
        # Reduce to the identity string: the pickle bytes are what gets hashed,
        # and the object is never actually reconstructed from them.
        pickler.save_reduce(str, (tokenizer_identity(obj),), obj=obj)

    pklregister(type(tokenizer))(_save_tokenizer_by_identity)
    return tokenizer_identity(tokenizer)


def to_messages(example):
    """Map one ReasonLite parquet row to TRL conversational format.

    `prompt` and `answer` already carry their own markers (the boxed-answer
    instruction and the <think>...</think> CoT), so they are copied verbatim.
    Missing keys raise KeyError (fail fast) rather than dropping the row.
    """
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def normalize_dataset(dataset, num_proc):
    """Map a loaded DatasetDict of ReasonLite parquet rows to `messages`.

    Pure of open-r1/torch so it unit-tests against a fake DatasetDict.
    remove_columns intersects with the columns actually present, so a renamed
    or extra metadata column never breaks the map.

    Each split is mapped with an explicit deterministic `new_fingerprint`
    (transform version tag + the split's source fingerprint). Without this,
    datasets falls back to a RANDOM fingerprint for the map output ("couldn't
    be hashed properly"), which makes SFTTrainer's downstream tokenize cache
    key differ per rank/run: all 16 ranks then tokenize the full split
    independently (16x64 procs on 64 CPUs -> throughput collapse) and no run
    reuses another's cache. A stable fingerprint lets rank0 tokenize once and
    the other ranks load the GPFS cache; reruns skip tokenize entirely.
    Version tag `v1` invalidates deliberately if to_messages changes.
    """
    out = {}
    for split, ds in dataset.items():
        remove = [c for c in _DROP_COLUMNS if c in ds.column_names]
        out[split] = ds.map(
            to_messages,
            remove_columns=remove,
            num_proc=num_proc,
            desc="Normalizing ReasonLite parquet to messages",
            new_fingerprint=f"reasonlite-messages-v1-{ds._fingerprint}",
        )
    return out


if __name__ == "__main__":
    import open_r1.sft as openr1_sft
    from open_r1.configs import ScriptArguments, SFTConfig
    from open_r1.sft import main
    from open_r1.utils import get_dataset as openr1_get_dataset
    from transformers import AutoTokenizer
    from trl import ModelConfig, TrlParser

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Give SFTTrainer's internal tokenize map a content-derived cache key before
    # any dataset work starts. The reducer is keyed on the tokenizer's concrete
    # class, so this throwaway instance only serves to resolve that class -- the
    # identity itself is recomputed from whichever instance SFTTrainer pickles.
    # Loading it here costs seconds; the miss it prevents costs ~12 minutes.
    _identity = register_stable_tokenizer_hash(
        AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, revision=model_args.model_revision
        )
    )
    print(f"[sft] stable tokenizer fingerprint registered: {_identity}", flush=True)

    # Intercept the get_dataset name that sft.py:main resolves (imported from
    # open_r1.utils into the sft module namespace) with a normalizing wrapper.
    # openr1_get_dataset is bound to the real loader here, so the wrapper can
    # never call itself -> no recursion. dataset_num_proc lives on SFTConfig
    # (training_args), not ScriptArguments.
    def _load_and_normalize(script_args):
        dataset = openr1_get_dataset(script_args)
        return normalize_dataset(dataset, training_args.dataset_num_proc)

    openr1_sft.get_dataset = _load_and_normalize

    main(script_args, training_args, model_args)
