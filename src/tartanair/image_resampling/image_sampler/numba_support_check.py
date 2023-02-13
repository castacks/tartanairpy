
# Special treatment for numba.
# On aarch64 (Jetson) platforms, we need to compile numba from source.
# However, numba depends on llvmlite which, in turn, depends on LLVM.
# It seems numba needs the LLVM to be compiled in a special way. It
# sounds too much work for now. 2022-12-26 by Yaoyu.
import platform
_U_ARCH = platform.uname().machine
_NUMBA_SUPPORTED_ARCHS = ('x86_64',)

def is_numba_supported():
    global _U_ARCH, _NUMBA_SUPPORTED_ARCHS
    return _U_ARCH in _NUMBA_SUPPORTED_ARCHS

def get_arch():
    global _U_ARCH
    return _U_ARCH