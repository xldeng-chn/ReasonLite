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

# Metadata columns dropped after building `messages` (SFTTrainer only needs
# the conversation; the rest are dataset provenance).
_DROP_COLUMNS = ["prompt", "answer", "expected_answer", "vote", "problem_source"]


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
    """
    present = set(next(iter(dataset.values())).column_names)
    remove = [c for c in _DROP_COLUMNS if c in present]
    return dataset.map(
        to_messages,
        remove_columns=remove,
        num_proc=num_proc,
        desc="Normalizing ReasonLite parquet to messages",
    )


if __name__ == "__main__":
    import open_r1.sft as openr1_sft
    from open_r1.configs import ScriptArguments, SFTConfig
    from open_r1.sft import main
    from open_r1.utils import get_dataset as openr1_get_dataset
    from trl import ModelConfig, TrlParser

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

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
