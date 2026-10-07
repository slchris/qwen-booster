"""A small localhost-only, single-request OpenAI chat API subset."""

import hashlib
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from .sampling import FIELDS, settings

MODEL_NAME = "qwen3.8-27b"


def validate_request(body):
    if not isinstance(body, dict):
        raise ValueError("request must be a JSON object")
    allowed = {"model", "messages", "stream", "max_tokens", "enable_thinking",
               "preserve_thinking", "reasoning_effort"} | FIELDS
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")
    if body.get("model", MODEL_NAME) != MODEL_NAME:
        raise ValueError(f"model must be {MODEL_NAME}")
    settings(body)
    if body.get("reasoning_effort", "xhigh") not in ("low", "medium", "xhigh"):
        raise ValueError("reasoning_effort must be low, medium or xhigh")
    for flag in ("stream", "enable_thinking", "preserve_thinking"):
        if flag in body and not isinstance(body[flag], bool):
            raise ValueError(f"{flag} must be boolean")
    limit = body.get("max_tokens", 256)
    if type(limit) is not int or not 1 <= limit <= 4096:
        raise ValueError("max_tokens must be an integer in [1, 4096]")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or not {"role", "content"} <= set(message)
                or set(message) - {"role", "content", "reasoning_content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str)):
            raise ValueError("messages require role (system/user/assistant) and text content")
        if "reasoning_content" in message and (message["role"] != "assistant"
                or not isinstance(message["reasoning_content"], str)):
            raise ValueError("reasoning_content requires an assistant message and text")
    return messages, limit, body.get("stream", False), body.get("enable_thinking", False)


def serve(args):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from .cli import local_model, environment
    from .runtime import Engine, HybridEngine, stream_lookup

    from .context import ContextBudget

    path, config = local_model(args.model)
    budget = ContextBudget.from_config(config, args.context)
    mx.set_memory_limit(args.memory_mb * 1_000_000)
    mx.set_cache_limit(args.allocator_cache_mb * 1_000_000)
    model, tokenizer = load(path)
    prefill_weights = {"enabled": False}
    if args.expanded_prefill:
        from .prefill_weights import expand_prefill_weights
        prefill_weights = expand_prefill_weights(model, args.memory_mb * 1_000_000)
    provenance = environment()
    provenance.update(model_path=str(path), model_config=config, prefill_weights=prefill_weights,
        model_files_sha256={name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                            for name in ("config.json", "model.safetensors.index.json", "tokenizer.json")
                            if (path / name).is_file()})
    engine = Engine(model, prefill_step_size=args.step, cache_bytes=args.cache_mb * 1_000_000,
                    active_memory_budget=args.memory_mb * 1_000_000)
    if getattr(args, "hybrid_cache", False):
        engine = HybridEngine(model, prefill_step_size=args.step,
                              cache_bytes=args.cache_mb * 1_000_000,
                              active_memory_budget=args.memory_mb * 1_000_000,
                              block_size=args.hybrid_block_size,
                              disk_root=args.cache_disk_dir,
                              disk_bytes=args.cache_disk_mb * 1_000_000,
                              disk_entries=args.cache_disk_entries)
    engine.check_memory()

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/v1/models":
                self.json_response(200, {"object": "list", "data": [
                    {"id": MODEL_NAME, "object": "model", "owned_by": "local"}]})
            elif self.path == "/health":
                self.json_response(200, {"status": "ok", "model": MODEL_NAME,
                    "context_limit": budget.service_limit, "native_context_limit": budget.native_limit,
                    "context_quality_validated": False, "environment": provenance,
                    "memory_budget_bytes": args.memory_mb * 1_000_000,
                    "allocator_cache_budget_bytes": args.allocator_cache_mb * 1_000_000,
                    "hybrid_cache": getattr(args, "hybrid_cache", False),
                    "hybrid_block_size": getattr(args, "hybrid_block_size", None),
                    "cache_disk_dir": getattr(args, "cache_disk_dir", None)})
            elif self.path == "/v1/cache/stats":
                self.json_response(200, engine.prefixes.stats())
            else:
                self.json_response(404, {"error": {"message": "unknown endpoint"}})

        def do_POST(self):
            if self.path == "/v1/cache/reset":
                engine.prefixes.clear()
                if getattr(engine, "disk", None) is not None:
                    engine.disk.clear()
                mx.clear_cache()
                self.json_response(200, engine.prefixes.stats())
                return
            if self.path != "/v1/chat/completions":
                self.json_response(404, {"error": {"message": "unknown endpoint"}})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("chunked request bodies are unsupported")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4_000_000:
                    raise ValueError("body must be between 1 and 4000000 bytes")
                self.connection.settimeout(30)
                body = json.loads(self.rfile.read(length))
                messages, limit, streaming, thinking = validate_request(body)
                sampling = settings(body)
                if args.lookup and (sampling["temperature"] != 0 or
                        sampling["presence_penalty"] != 0 or sampling["repetition_penalty"] != 1):
                    raise ValueError("lookup supports greedy sampling without penalties only")
                tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                            add_generation_prompt=True, enable_thinking=thinking,
                            preserve_thinking=body.get("preserve_thinking", True),
                            reasoning_effort=body.get("reasoning_effort", "xhigh"))
                budget.admit(len(tokens), limit)
            except (ValueError, TypeError, TimeoutError) as error:
                self.json_response(400, {"error": {"message": str(error), "type": "invalid_request_error"}})
                return
            iterator = None
            headers_sent = False
            try:
                started = time.perf_counter()
                mx.reset_peak_memory()
                cache = engine.prepare(tokens)
                prepare_seconds = time.perf_counter() - started
                if args.lookup:
                    iterator = stream_lookup(model, tokenizer, tokens, cache, max_tokens=limit)
                else:
                    from .sampling import native_options
                    iterator = stream_generate(model, tokenizer, tokens[-1:], max_tokens=limit,
                        prompt_cache=cache, **native_options(sampling, tokens, limit))
                identifier = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())

                def payload(choices, chunk=False):
                    return {"id": identifier, "object": "chat.completion.chunk" if chunk else "chat.completion",
                            "created": created, "model": MODEL_NAME, "choices": choices}

                def send_event(value):
                    self.wfile.write(("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()

                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    headers_sent = True
                    send_event(payload([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}], True))
                from .reasoning import ReasoningParser
                parser = ReasoningParser(thinking)
                text = reasoning = ""
                generated = []
                first_token = last_token = None
                for response in iterator:
                    engine.check_memory()
                    now = time.perf_counter()
                    first_token = now if first_token is None else first_token
                    last_token = now
                    if not args.lookup:
                        generated.append(response.token)
                    delta = parser.feed(response.text, final=response.finish_reason is not None)
                    text += delta.get("content", "")
                    reasoning += delta.get("reasoning_content", "")
                    if streaming:
                        send_event(payload([{"index": 0, "delta": delta,
                                             "finish_reason": response.finish_reason}], True))
                committed = False
                if not args.lookup:
                    iterator.close()
                    committed = engine.commit(tokens + generated, cache)
                timings = {"prompt_tokens": len(tokens), "cached_tokens": engine.last_cached_tokens,
                    "new_prompt_tokens": len(tokens) - engine.last_cached_tokens,
                    "prepare_seconds": prepare_seconds,
                    "first_token_seconds": first_token - started,
                    "generation_tokens": response.generation_tokens,
                    "decode_seconds": last_token - first_token,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "active_mlx_bytes": mx.get_active_memory(),
                    "allocator_cache_bytes": mx.get_cache_memory(),
                    "prefix_store": engine.prefixes.stats(), "completion_cache_committed": committed}
                if streaming:
                    telemetry = payload([], True)
                    telemetry["timings"] = timings
                    send_event(telemetry)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    result = payload([{"index": 0, "message": {"role": "assistant", "content": text},
                                       "finish_reason": response.finish_reason}])
                    if reasoning:
                        result["choices"][0]["message"]["reasoning_content"] = reasoning
                    result["usage"] = {"prompt_tokens": len(tokens), "completion_tokens": response.generation_tokens,
                                       "total_tokens": len(tokens) + response.generation_tokens}
                    result["timings"] = timings
                    self.json_response(200, result)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except MemoryError as error:
                engine.prefixes.clear()
                mx.clear_cache()
                self.log_error("memory budget exceeded: %s", error)
                if not headers_sent:
                    self.json_response(503, {"error": {"message": str(error), "type": "memory_budget_exceeded"}})
                else:
                    self.close_connection = True
            except Exception as error:
                self.log_error("generation failed: %s", error)
                if not headers_sent:
                    self.json_response(500, {"error": {"message": "generation failed"}})
                else:
                    self.close_connection = True
            finally:
                if iterator is not None:
                    iterator.close()
                mx.synchronize()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.timeout = 1
    print(f"Serving {MODEL_NAME} at http://127.0.0.1:{args.port}/v1 "
          f"(one request at a time, lookup={args.lookup}, "
          f"hybrid_cache={getattr(args, 'hybrid_cache', False)})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
