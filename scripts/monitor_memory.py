"""Sample an existing benchmark process; macOS RSS excludes some GPU allocations."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.pid <= 0 or args.interval <= 0:
        parser.error("pid and interval must be positive")
    def read(*command):
        return subprocess.check_output(command, text=True).strip()
    identity = read("ps", "-p", str(args.pid), "-o", "lstart=", "-o", "command=")
    with Path(args.output).open("x") as output:
        output.write(json.dumps({"type": "metadata", "pid": args.pid, "identity": identity,
            "scope": "process RSS plus system swap; not total GPU allocation; sampling may miss peaks"}) + "\n")
        while True:
            try:
                current = read("ps", "-p", str(args.pid), "-o", "lstart=", "-o", "command=")
                if current != identity:
                    break
                row = {"type": "sample", "unix_time": time.time(),
                       "rss_bytes": int(read("ps", "-p", str(args.pid), "-o", "rss=")) * 1024,
                       "swap": read("sysctl", "-n", "vm.swapusage")}
            except subprocess.CalledProcessError:
                break
            output.write(json.dumps(row) + "\n")
            output.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
