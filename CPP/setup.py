from setuptools import setup, Extension
import pybind11
import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# Detect fmt library (libfmt) include / link paths
# Try pkg-config first, fall back to empty (user may set CXXFLAGS / LDFLAGS).
# ---------------------------------------------------------------------------
print("pkg-config --cflags fmt")
def pkg_config(flag, lib):
    try:
        out = subprocess.check_output(["pkg-config", flag, lib],
                                      stderr=subprocess.DEVNULL)
        return out.decode().split()
    except Exception:
        return []

fmt_cflags  = pkg_config("--cflags", "fmt")
fmt_ldflags = pkg_config("--libs",   "fmt")


openmp_cflags  = pkg_config("--cflags", "openmp")
openmp_ldflags = pkg_config("--libs",   "openmp")
# Strip -I prefix for include_dirs, -L for library_dirs, -l for libraries
extra_include = [f[2:] for f in fmt_cflags  if f.startswith("-I")]
extra_libdirs  = [f[2:] for f in fmt_ldflags if f.startswith("-L")]
extra_libs     = [f[2:] for f in fmt_ldflags if f.startswith("-l")]

print(extra_include)
print(extra_libdirs)
print(extra_libs)
if not extra_libs:
    # pkg-config not available — assume libfmt is in the default search path
    extra_libs = ["fmt"]

class EigenBuildExt(build_ext):
    def build_extensions(self):
        # 1. Define Eigen version and path
        eigen_dir = os.path.abspath("eigen")

        # 3. Add Eigen to include paths for all extensions
        for ext in self.extensions:
            ext.include_dirs.append(eigen_dir)
            
        super().build_extensions()



# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------
ext = Extension(
    name="fermionic_ops",                     # import name in Python
    sources=["m1.cpp"],
    include_dirs=[
        pybind11.get_include(),               # pybind11 headers
        *extra_include,
    ],
    library_dirs=extra_libdirs,
    libraries=extra_libs,
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language="c++",
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="fermionic_ops",
    version="0.1.0",
    author="",
    description="Fermionic operator loop (opLoop) exposed via pybind11",
    ext_modules=[ext],
    zip_safe=False,
    python_requires=">=3.8",
)
