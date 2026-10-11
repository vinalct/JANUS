"""Check the persistent-container exec path before the expensive Spark suites."""

from __future__ import annotations

import os
import pwd
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.support.spark_sessions import build_iceberg_session


def main() -> int:
    account = pwd.getpwuid(os.getuid())
    assert Path(account.pw_dir).is_absolute(), f"Account home must be absolute: {account.pw_dir}"
    with TemporaryDirectory(prefix="janus-spark-startup-") as directory:
        session = build_iceberg_session("janus-container-spark-startup", Path(directory))
        try:
            user_home = session.sparkContext._jvm.java.lang.System.getProperty("user.home")
            assert Path(user_home).is_absolute(), f"JVM user.home must be absolute: {user_home}"
            assert session.sql("SELECT 1 AS ready").first().ready == 1
            print(
                f"OK: uid {account.pw_uid} resolves as {account.pw_name}; "
                f"JVM home {user_home}; Spark SQL executed"
            )
        finally:
            session.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
