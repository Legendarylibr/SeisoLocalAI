"""Platform resource-limit helpers for subprocess code execution.

Naming note: this is not an OS sandbox — only rlimits / best-effort process
isolation. Prefer ``SEISO_ALLOW_CODE_EXEC=false`` (especially with remote access).
"""

from __future__ import annotations

import os
import subprocess

from forge.tools.code_exec_policy import (
    _MAX_OPEN_FDS,
    _MAX_RSS_BYTES,
    _TIMEOUT_SEC,
)

_MAX_NPROC = 32
_MAX_FSIZE_BYTES = 10 * 1024 * 1024


def subprocess_limits() -> None:
    """Apply resource limits in the code-exec child process (Unix only)."""
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_TIMEOUT_SEC, _TIMEOUT_SEC + 5))
        resource.setrlimit(resource.RLIMIT_AS, (_MAX_RSS_BYTES, _MAX_RSS_BYTES))
        if hasattr(resource, "RLIMIT_NOFILE"):
            resource.setrlimit(resource.RLIMIT_NOFILE, (_MAX_OPEN_FDS, _MAX_OPEN_FDS))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (_MAX_NPROC, _MAX_NPROC))
        if hasattr(resource, "RLIMIT_FSIZE"):
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (_MAX_FSIZE_BYTES, _MAX_FSIZE_BYTES)
            )
    except (OSError, ValueError):
        pass


def windows_job_limits() -> int | None:
    """Return a Windows job handle with process memory cap, or None if unavailable."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = _MAX_RSS_BYTES
        JobObjectExtendedLimitInformation = 9
        kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        return int(job)
    except (AttributeError, OSError, ValueError):
        return None


def assign_windows_job(proc: subprocess.Popen[str], job_handle: int | None) -> None:
    if os.name != "nt" or not job_handle:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.AssignProcessToJobObject(  # type: ignore[attr-defined]
            job_handle,
            int(proc._handle),  # noqa: SLF001
        )
    except (AttributeError, OSError, ValueError):
        pass
