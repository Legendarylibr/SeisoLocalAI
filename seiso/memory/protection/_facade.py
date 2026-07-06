"""Lazy access to the protection package facade (monkeypatch-friendly)."""


def protection():
    import seiso.memory.protection as prot

    return prot
