"""Lazy access to the model_pool package facade (monkeypatch-friendly)."""


def model_pool():
    import seiso.inference.model_pool as mp

    return mp
