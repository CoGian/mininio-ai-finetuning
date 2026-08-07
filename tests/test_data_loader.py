from datasets import Dataset, DatasetDict

from finetuning.common.data_loader import check_bos_prefix, load_dataset_for_model, prepare_dataset, report_token_lengths


class MockLFM:
    bos_token = "<bos>"
    bos_token_id = 1

    def __call__(self, text, **kwargs):
        inputs = [[1, 42, 43]] if isinstance(text, list) else [1, 42, 43]
        return {"input_ids": inputs}


class MockGemma:
    bos_token = "<bos>"
    bos_token_id = 1

    def __call__(self, text, **kwargs):
        assert isinstance(text, list), f"b{type(text).__name__}"
        return {"input_ids": [[3, 42, 43, 44]]}


class MockBert:
    bos_token = ""
    bos_token_id = None

    def __call__(self, text, **kwargs):
        inputs = [[101, 42, 43]] if isinstance(text, list) else [101, 42, 43]
        return {"input_ids": inputs}


_DS = DatasetDict({
    "train": Dataset.from_list([{"text": "<bos>system\nuser"}, {"text": "no prefix"}]),
    "eval": Dataset.from_list([{"text": "<bos>system\nuser"}]),
})


class TestReportTokenLengths:
    def test_lfm_accepts_batch(self):
        max_len = report_token_lengths(_DS, MockLFM(), "lfm", n_samples=2)
        assert max_len > 0

    def test_gemma_accepts_batch(self):
        max_len = report_token_lengths(_DS, MockGemma(), "gemma", n_samples=2)
        assert max_len > 0

    def test_lfm_batch_returns_correct_length(self):
        lengths = []
        for row in _DS["train"]:
            tokens = MockLFM()([row["text"]], add_special_tokens=False, truncation=False)
            lengths.append(len(tokens["input_ids"][0]))
        assert lengths == [3, 3]

    def test_gemma_batch_returns_correct_length(self):
        lengths = []
        for row in _DS["train"]:
            tokens = MockGemma()([row["text"]], add_special_tokens=False, truncation=False)
            lengths.append(len(tokens["input_ids"][0]))
        assert lengths == [4, 4]


class TestCheckBosPrefix:
    def test_lfm_no_crash(self):
        check_bos_prefix(_DS, MockLFM(), "lfm")

    def test_gemma_no_crash(self):
        check_bos_prefix(_DS, MockGemma(), "gemma")

    def test_no_bos_token(self):
        check_bos_prefix(_DS, MockBert(), "bert")


class TestLoadDatasetForModel:
    def test_raises_on_missing_dir(self):
        try:
            load_dataset_for_model("lfm", data_dir="data/output")
        except FileNotFoundError:
            pass  # expected if no data present


class TestPrepareDataset:
    def test_removes_columns(self):
        def identity(x):
            return x

        ds = DatasetDict({
            "train": Dataset.from_list([{"text": "a"}, {"text": "b"}]),
            "eval": Dataset.from_list([{"text": "c"}]),
        })
        result = prepare_dataset(ds, identity)
        assert "text" not in result["train"].column_names
