"""Put every vendored jar on the JVM's launch classpath, whichever Spark module runs first."""

from __future__ import annotations

import os
from pathlib import Path

IVY_JARS_DIR = Path(__file__).resolve().parent.parent / "data" / "metadata" / "ivy" / "jars"
SUBMIT_ARGS_ENV = "PYSPARK_SUBMIT_ARGS"
# PySpark appends this verbatim to the spark-submit command, and it must end in `pyspark-shell`.
PYSPARK_SHELL_ARG = "pyspark-shell"


def _seeded_jars() -> list[str]:
    return sorted(str(jar) for jar in IVY_JARS_DIR.glob("*.jar"))


def _install_launch_classpath() -> None:
    if os.environ.get(SUBMIT_ARGS_ENV):
        return
    jars = _seeded_jars()
    if not jars:
        return
    os.environ[SUBMIT_ARGS_ENV] = f"--jars {','.join(jars)} {PYSPARK_SHELL_ARG}"


_install_launch_classpath()
