"""Test-suite bootstrap.

The unit tests must run while the persistent daemon owns the 8 TPU chips, so they use
the CPU backend forced to 8 devices. That still exercises the real 8-way ``tp`` mesh,
the ``NamedSharding`` specs and every collective, just at toy sizes.

Set ``GEMMA4_TEST_PLATFORM=tpu`` to run the same tests on real silicon (stop the daemon
first: ``bash serve.sh stop``).
"""

import os

platform = os.environ.get("GEMMA4_TEST_PLATFORM", "cpu")
os.environ["JAX_PLATFORMS"] = platform
if platform == "cpu":
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_force_host_platform_device_count=8").strip()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
