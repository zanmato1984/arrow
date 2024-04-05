# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys

import pytest

import pyarrow as pa


pytestmark = pytest.mark.gdb

here = os.path.dirname(os.path.abspath(__file__))

# The GDB script may be found in the source tree (if available)
# or in another location given by the ARROW_GDB_SCRIPT environment variable.
gdb_script = (os.environ.get('ARROW_GDB_SCRIPT') or
              os.path.join(here, "../../../cpp/gdb_arrow.py"))

gdb_command = ["gdb", "--nx"]


def environment_for_gdb():
    env = {}
    for var in ['PATH', 'LD_LIBRARY_PATH']:
        try:
            env[var] = os.environ[var]
        except KeyError:
            pass
    return env


@lru_cache()
def is_gdb_available():
    try:
        # Try to use the same arguments as in GdbSession so that the
        # same error return gets propagated.
        proc = subprocess.run(gdb_command + ["--version"],
                              env=environment_for_gdb(), bufsize=0,
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return False
    return proc.returncode == 0


@lru_cache()
def python_executable():
    path = shutil.which("python3")
    assert path is not None, "Couldn't find python3 executable"
    return path


def skip_if_gdb_unavailable():
    if not is_gdb_available():
        pytest.skip("gdb command unavailable")


def skip_if_gdb_script_unavailable():
    if not os.path.exists(gdb_script):
        pytest.skip("gdb script not found")


class GdbSession:
    proc = None
    verbose = True

    def __init__(self, *args, **env):
        # Let stderr through to let pytest display it separately on errors
        gdb_env = environment_for_gdb()
        gdb_env.update(env)
        self.proc = subprocess.Popen(gdb_command + list(args),
                                     env=gdb_env, bufsize=0,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.last_stdout = []
        self.last_stdout_line = b""

    def wait_until_ready(self):
        """
        Record output until the gdb prompt displays.  Return recorded output.
        """
        # TODO: add timeout?
        while (not self.last_stdout_line.startswith(b"(gdb) ") and
               self.proc.poll() is None):
            block = self.proc.stdout.read(4096)
            if self.verbose:
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
            block, sep, last_line = block.rpartition(b"\n")
            if sep:
                self.last_stdout.append(self.last_stdout_line)
                self.last_stdout.append(block + sep)
                self.last_stdout_line = last_line
            else:
                assert block == b""
                self.last_stdout_line += last_line

        if self.proc.poll() is not None:
            raise IOError("gdb session terminated unexpectedly")

        out = b"".join(self.last_stdout).decode('utf-8')
        self.last_stdout = []
        self.last_stdout_line = b""
        return out

    def issue_command(self, line):
        line = line.encode('utf-8') + b"\n"
        if self.verbose:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def run_command(self, line):
        self.issue_command(line)
        return self.wait_until_ready()

    def print_value(self, expr):
        """
        Ask gdb to print the value of an expression and return the result.
        """
        out = self.run_command(f"p {expr}")
        out, n = re.subn(r"^\$\d+ = ", "", out)
        # assert n == 1, out
        # gdb may add whitespace depending on result width, remove it
        return out.strip()

    def print_value_type(self, expr):
        """
        Ask gdb to print the value of an expression and return the result.
        """
        out = self.run_command(f"whatis {expr}")
        # gdb may add whitespace depending on result width, remove it
        return out.strip()

    def select_frame(self, func_name):
        """
        Select the innermost frame with the given function name.
        """
        # Ideally, we would use the "frame function" command,
        # but it's not available on old GDB versions (such as 8.1.1),
        # so instead parse the stack trace for a matching frame number.
        out = self.run_command("info stack")
        pat = r"(?mi)^#(\d+)\s+.* in " + re.escape(func_name) + r"\b"
        m = re.search(pat, out)
        if m is None:
            pytest.fail(f"Could not select frame for function {func_name}")

        frame_num = int(m[1])
        out = self.run_command(f"frame {frame_num}")
        assert f"in {func_name}" in out

    def join(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.stdout.close()  # avoid ResourceWarning
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def __del__(self):
        self.join()


@pytest.fixture(scope='session')
def gdb():
    skip_if_gdb_unavailable()
    gdb = GdbSession("-q", python_executable())
    try:
        gdb.wait_until_ready()
        gdb.run_command("set confirm off")
        gdb.run_command("set print array-indexes on")
        # Make sure gdb formatting is not terminal-dependent
        gdb.run_command("set width unlimited")
        gdb.run_command("set charset UTF-8")
        yield gdb
    finally:
        gdb.join()


@pytest.fixture(scope='session')
def gdb_arrow(gdb):
    if 'deb' not in pa.cpp_build_info.build_type:
        pytest.skip("Arrow C++ debug symbols not available")

    skip_if_gdb_script_unavailable()
    gdb.run_command(f"source {gdb_script}")

    lib_path_var = 'PATH' if sys.platform == 'win32' else 'LD_LIBRARY_PATH'
    lib_path = os.environ.get(lib_path_var)
    if lib_path:
        # GDB starts the inferior process in a pristine shell, need
        # to propagate the library search path to find the Arrow DLL
        gdb.run_command(f"set env {lib_path_var} {lib_path}")

    code = "from pyarrow.lib import _gdb_test_session; _gdb_test_session()"
    out = gdb.run_command(f"run -c '{code}'")
    assert ("Trace/breakpoint trap" in out or
            "received signal" in out), out
    gdb.select_frame("arrow::gdb::TestSession")
    return gdb

def check_stack_repr(gdb, expr, expected):
    """
    Check printing a stack-located value.
    """
    s = gdb.print_value(expr)
    if isinstance(expected, re.Pattern):
        assert expected.match(s), s
    else:
        assert s == expected


def check_heap_repr(gdb, expr, expected):
    """
    Check printing a heap-located value, given its address.
    """
    s = gdb.print_value(f"*{expr}")
    # GDB may prefix the value with an address or type specification
    if s != expected:
        assert s.endswith(f" {expected}")

def test_scalars_stack(gdb_arrow):
    sys.stderr.write("!!!Debug!!!")

    sys.stderr.write(f"scalar addr: {gdb_arrow.print_value('&binary_scalar_null')}\n")
    sys.stderr.write(f"scalar addr: {gdb_arrow.print_value('&binary_scalar_unallocated')}\n")
    sys.stderr.write(f"scalar addr: {gdb_arrow.print_value('&binary_scalar_empty')}\n")

    sys.stderr.write(f"type type: {gdb_arrow.print_value_type('binary_scalar_null.type')}\n")
    sys.stderr.write(f"type addr: {gdb_arrow.print_value('binary_scalar_null.type.get()')}\n")
    sys.stderr.write(f"type: {gdb_arrow.print_value('binary_scalar_null.type')}\n")
    sys.stderr.write(f"type deref: {gdb_arrow.print_value('*(binary_scalar_null.type)')}\n")

    sys.stderr.write(f"type type: {gdb_arrow.print_value_type('binary_scalar_unallocated.type')}\n")
    sys.stderr.write(f"type addr: {gdb_arrow.print_value('binary_scalar_unallocated.type.get()')}\n")
    sys.stderr.write(f"type: {gdb_arrow.print_value('binary_scalar_unallocated.type')}\n")
    sys.stderr.write(f"type deref: {gdb_arrow.print_value('*(binary_scalar_unallocated.type)')}\n")

    sys.stderr.write(f"type type: {gdb_arrow.print_value_type('binary_scalar_empty.type')}\n")
    sys.stderr.write(f"type addr: {gdb_arrow.print_value('binary_scalar_empty.type.get()')}\n")
    sys.stderr.write(f"type: {gdb_arrow.print_value('binary_scalar_empty.type')}\n")
    sys.stderr.write(f"type deref: {gdb_arrow.print_value('*(binary_scalar_empty.type)')}\n")

    # sys.stderr.write(f"value type: {gdb_arrow.print_value_type('binary_scalar_empty.value')}\n")
    # sys.stderr.write(f"value addr: {gdb_arrow.print_value('binary_scalar_empty.value.get()')}\n")
    # sys.stderr.write(f"value: {gdb_arrow.print_value('binary_scalar_empty.value')}\n")
    # sys.stderr.write(f"value deref: {gdb_arrow.print_value('*(binary_scalar_empty.value)')}\n")
    # sys.stderr.write(f"scalar type: {gdb_arrow.print_value_type('binary_scalar_empty')}\n")
    # sys.stderr.write(f"scalar addr: {gdb_arrow.print_value('&binary_scalar_empty')}\n")
    # sys.stderr.write(f"scalar: {gdb_arrow.print_value('binary_scalar_empty')}\n")
    sys.stderr.write("!!!Debug!!!")

    check_stack_repr(
        gdb_arrow, "binary_scalar_empty",
        'arrow::BinaryScalar of size 0, value ""')
