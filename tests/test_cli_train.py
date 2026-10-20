"""Regression tests for publishing a trained model from the CLI."""

from purescribe import cli


def test_train_refuses_existing_model_before_reading_corpus(tmp_path, capsys):
    model = tmp_path / "model.psm"
    original = b"existing model"
    model.write_bytes(original)

    result = cli.main(["train", str(tmp_path / "missing-corpus"), "-o", str(model)])

    assert result == 2
    assert model.read_bytes() == original
    assert "output already exists" in capsys.readouterr().err
