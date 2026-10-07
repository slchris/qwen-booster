"""Exercise both real-model API modes, then stop only the servers we started."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    results = Path("results")
    results.mkdir(exist_ok=True)
    report_path = results / "api-smoke.json"
    if report_path.exists():
        raise SystemExit(f"Refusing to overwrite {report_path}")
    reports = []
    for lookup in (False, True):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [sys.executable, "-m", "qwen_booster.cli", "serve", "--port", str(port)]
        if lookup:
            command.append("--lookup")
        with (results / f"api-{'lookup' if lookup else 'native'}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log,
                        env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("server exited during startup; inspect API log")
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=1) as response:
                            assert json.load(response)["status"] == "ok"
                        break
                    except urllib.error.URLError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("server did not start within 60 seconds")
                        time.sleep(0.1)

                def request(body):
                    return urllib.request.urlopen(urllib.request.Request(
                        base + "/v1/chat/completions", data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"}), timeout=60)

                body = {"model": "qwen3.8-27b", "messages": [
                    {"role": "user", "content": "用一句话解释二分查找。"}], "max_tokens": 32, "temperature": 0}
                with request(body) as response:
                    reference = json.load(response)
                expected = reference["choices"][0]["message"]["content"]
                assert expected and reference["usage"]["prompt_tokens"] > 1
                with request({**body, "stream": True}) as response:
                    fragments = []
                    finished = False
                    done = False
                    for line in response:
                        if line.strip() == b"data: [DONE]":
                            done = True
                        elif line.startswith(b"data: "):
                            choices = json.loads(line[6:])["choices"]
                            if not choices:
                                continue
                            chunk = choices[0]
                            fragments.append(chunk["delta"].get("content", ""))
                            finished |= chunk["finish_reason"] in ("length", "stop")
                    assert done and finished and "".join(fragments) == expected
                other = {**body, "messages": [{"role": "user", "content": "Write Python code to add two integers."}]}
                with request(other) as response:
                    assert json.load(response)["choices"][0]["message"]["content"]
                # Disconnect after actual content; the next request must still work.
                with request({**other, "stream": True, "max_tokens": 4096}) as response:
                    for line in response:
                        if line.startswith(b"data: "):
                            chunk = json.loads(line[6:])
                            if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                                break
                with request(body) as response:
                    assert json.load(response)["choices"][0]["message"]["content"] == expected
                try:
                    request({**body, "temperature": -0.5})
                    raise AssertionError("unsupported sampling was silently accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
                    assert json.load(response)["data"][0]["id"] == body["model"]
                reports.append({"lookup": lookup, "passed": True, "response": expected,
                    "checks": ["health", "models", "JSON", "SSE", "warm-cache equality",
                               "branch isolation", "disconnect recovery", "unsupported sampling rejection"]})
                print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
            finally:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    report_path.write_text(json.dumps({"passed": True, "modes": reports}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
