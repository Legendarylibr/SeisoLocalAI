"""Linux CUDA runtime and toolkit repair helpers."""

from __future__ import annotations


def test_pip_nvidia_cuda_lib_dirs_prioritizes_cu12_runtime(tmp_path, monkeypatch):
    from seiso import platform as plat

    site = tmp_path / "site-packages"
    nvidia = site / "nvidia"
    (nvidia / "cu13" / "lib").mkdir(parents=True)
    (nvidia / "cu13" / "lib" / "libcudart.so.13").write_bytes(b"")
    (nvidia / "cuda_runtime" / "lib").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "lib" / "libcudart.so.12").write_bytes(b"")
    (nvidia / "cublas" / "lib").mkdir(parents=True)
    (nvidia / "cublas" / "lib" / "libcublas.so.12").write_bytes(b"")

    monkeypatch.setattr(plat.site, "getsitepackages", lambda: [str(site)])
    monkeypatch.setattr(plat.site, "getusersitepackages", lambda: "")

    dirs = plat.pip_nvidia_cuda_lib_dirs()
    assert dirs
    assert "cuda_runtime" in dirs[0]
    assert any("cu13" in d for d in dirs)


def test_cu12_runtime_installed_detects_libcudart12(tmp_path, monkeypatch):
    from seiso import platform as plat

    site = tmp_path / "site-packages"
    lib = site / "nvidia" / "cuda_runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcudart.so.12").write_bytes(b"")

    monkeypatch.setattr(plat.site, "getsitepackages", lambda: [str(site)])
    monkeypatch.setattr(plat.site, "getusersitepackages", lambda: "")
    assert plat.cu12_runtime_installed() is True


def test_ensure_cu12_runtime_skips_without_nvidia(monkeypatch):
    from seiso import platform as plat

    monkeypatch.setattr(plat, "cu12_runtime_installed", lambda: False)
    monkeypatch.setattr("seiso.security.nvidia_boundary.nvidia_smi_visible", lambda: False)
    assert plat.ensure_cu12_runtime_packages(auto_install=True) is True


def test_repair_linux_cuda_stack_skips_non_linux(monkeypatch):
    from seiso import platform as plat

    monkeypatch.setattr(plat.platform, "system", lambda: "Darwin")
    report = plat.repair_linux_cuda_stack()
    assert report.get("skipped") is True


def test_repair_cuda_ptxas_noop_when_compatible(monkeypatch):
    from seiso import platform as plat

    monkeypatch.setattr(
        "seiso.kernels.cuda_env.cuda_toolkit_status",
        lambda: {"ptxas_compatible": True},
    )
    assert plat.repair_cuda_ptxas_toolkit(auto_install=True) is True
