"""Linux pidfd APIs for Python builds without the optional native bindings.

No numeric-PID signalling fallback. libc wrappers select the architecture's
ABI; deliberately do not guess syscall numbers on unsupported libc/platforms.
The caller must still authenticate trial and process identity around delivery.
"""
import ctypes
import errno
import os
import signal
import sys


def _libc_call(name, argtypes, *args):
    if sys.platform != 'linux':
        raise OSError(errno.ENOSYS, 'Linux pidfd required')
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        call = getattr(libc, name)
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, 'libc lacks ' + name) from exc
    call.argtypes, call.restype = argtypes, ctypes.c_int
    result = call(*args)
    if result < 0:
        error = ctypes.get_errno()
        # Preserve errno/subclasses, including EBADF, ESRCH, ENOSYS, EPERM.
        # Only ESRCH may be treated as an exited process by the caller.
        raise OSError(error, name + ': ' + os.strerror(error))
    return result


def pidfd_open(pid):
    native = getattr(os, 'pidfd_open', None)
    if callable(native):
        return native(pid, 0)
    return _libc_call('pidfd_open', (ctypes.c_int, ctypes.c_uint), pid, 0)


def pidfd_send_signal(fd, sig):
    native = getattr(signal, 'pidfd_send_signal', None)
    if callable(native):
        return native(fd, sig, None, 0)
    return _libc_call('pidfd_send_signal',
                      (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint),
                      fd, sig, None, 0)


def check_signal_support():
    """Probe the actual kernel/permission path, with no delivered signal."""
    try:
        fd = pidfd_open(os.getpid())
        try:
            pidfd_send_signal(fd, 0)
        finally:
            os.close(fd)
    except OSError as exc:
        raise RuntimeError('PIDFD_SIGNAL_SUPPORT_REQUIRED: ' + str(exc)) from exc
    return dict(pidfd_open='python' if callable(getattr(os, 'pidfd_open', None)) else 'libc',
                pidfd_send_signal='python' if callable(getattr(signal, 'pidfd_send_signal', None)) else 'libc',
                self_signal_zero=True)
