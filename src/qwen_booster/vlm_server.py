"""Isolated experimental MLX-VLM server with explicit context and memory budgets."""

import argparse
import math
import os
import sys
from pathlib import Path


def disk_cache_environment(directory, budget_mb, memory_cache_mb):
    """Require explicit storage and a bounded capacity before persisting prompts."""
    if budget_mb < 0 or bool(directory) != (budget_mb > 0):
        raise ValueError('disk cache requires both --disk-cache-dir and positive --disk-cache-mb')
    if not directory:
        return {'APC_DISK_ENABLED': '0'}
    if memory_cache_mb <= 0:
        raise ValueError('disk cache requires positive --cache-mb to enable APC')
    path = Path(directory).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError('disk cache path must be a directory')
    return {'APC_DISK_ENABLED': '1', 'APC_DISK_PATH': str(path),
            'APC_DISK_MAX_GB': str(budget_mb * 1_000_000 / (1 << 30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--fused-prefill", action="store_true",
                        help="Experimental 27B fused full-attention prefill; defaults off")
    parser.add_argument("--block", type=int, default=3)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--memory-mb", type=int, default=64000)
    parser.add_argument("--allocator-cache-mb", type=int, default=2000)
    parser.add_argument("--cache-mb", type=int, default=0,
                        help="Experimental in-memory APC budget; 0 disables prefix reuse")
    parser.add_argument("--disk-cache-dir",
                        help="Experimental: persist complete prompt-cache snapshots in this local directory")
    parser.add_argument("--disk-cache-mb", type=int, default=0,
                        help="Explicit disk cache capacity in decimal MB; requires --disk-cache-dir")
    parser.add_argument("--token-timeout", type=float, default=7200,
                        help="Seconds to wait for a generated token, including cold prefill (default: 7200)")
    parser.add_argument("--expanded-prefill", action="store_true",
                        help="Experimental prefill-only MLP expansion; about 34 GB additional memory")
    parser.add_argument("--release-finished-mtp", action="store_true",
                        help="Experimental cleanup of completed MTP batches before the next request")
    parser.add_argument("--normalize-single-warm-cache", action="store_true",
                        help="Experimental 27B non-MTP singleton APC cache normalization")
    parser.add_argument("--credit-live-warm-cache", action="store_true",
                        help="Experimental Flash-Next non-MTP restored-cache memory credit")
    parser.add_argument("--borrow-single-checkpoint-row", action="store_true",
                        help="Experimental Flash-Next non-MTP deferred checkpoint row copy")
    parser.add_argument("--borrow-flash-mtp-checkpoint-row", action="store_true",
                        help="Experimental Flash-Next MTP deferred checkpoint row copy; defaults off")
    parser.add_argument("--credit-flash-mtp-live-warm-cache", action="store_true",
                        help="Experimental Flash MTP restored-cache credit; requires borrowed checkpoint rows")
    parser.add_argument("--recheck-flash-mtp-cache-admission", action="store_true",
                        help="Experimental Flash MTP post-eviction admission recheck; requires live-cache credit")
    parser.add_argument("--recheck-flash-mtp-cache-restore", action="store_true",
                        help="Experimental Flash MTP post-eviction restore recheck; requires store recheck")
    parser.add_argument("--borrow-27b-checkpoint-row", action="store_true",
                        help="Experimental 27B singleton deferred checkpoint row copy")
    parser.add_argument("--credit-27b-live-warm-cache", action="store_true",
                        help="Experimental 27B singleton restored-cache memory credit")
    parser.add_argument("--recheck-27b-cache-admission", action="store_true",
                        help="Experimental bounded exact-cache admission recheck after eviction")
    parser.add_argument("--recheck-27b-first-store", action="store_true",
                        help="Experimental empty-cache store recheck; requires --recheck-27b-cache-admission")
    args = parser.parse_args()
    try:
        disk_environment = disk_cache_environment(args.disk_cache_dir, args.disk_cache_mb, args.cache_mb)
    except ValueError as error:
        parser.error(str(error))
    if (not 1 <= args.port <= 65535 or args.block < 2
            or min(args.memory_mb, args.allocator_cache_mb) <= 0 or args.cache_mb < 0
            or not math.isfinite(args.token_timeout) or args.token_timeout <= 0):
        parser.error("invalid port, block size, memory budget or token timeout")
    from .models import resolve_local_model
    from .context import ContextBudget

    path, config, profile = resolve_local_model(args.model)
    ContextBudget.from_config(config, args.context)
    if args.fused_prefill and profile.id != 'qwen3.8-27b':
        parser.error("--fused-prefill requires the 27B model")
    if args.normalize_single_warm_cache and (profile.id != 'qwen3.8-27b' or args.draft):
        parser.error("--normalize-single-warm-cache requires 27B without a draft")
    if args.credit_live_warm_cache and (profile.id != 'qwen3.8-flash-next' or args.draft):
        parser.error("--credit-live-warm-cache requires Flash-Next without a draft")
    if args.borrow_single_checkpoint_row and (profile.id != 'qwen3.8-flash-next' or args.draft):
        parser.error("--borrow-single-checkpoint-row requires Flash-Next without a draft")
    if args.borrow_flash_mtp_checkpoint_row and (profile.id != 'qwen3.8-flash-next' or not args.draft):
        parser.error("--borrow-flash-mtp-checkpoint-row requires Flash-Next with a draft")
    if args.credit_flash_mtp_live_warm_cache and not args.borrow_flash_mtp_checkpoint_row:
        parser.error("--credit-flash-mtp-live-warm-cache requires --borrow-flash-mtp-checkpoint-row")
    if args.recheck_flash_mtp_cache_admission and not args.credit_flash_mtp_live_warm_cache:
        parser.error("--recheck-flash-mtp-cache-admission requires --credit-flash-mtp-live-warm-cache")
    if args.recheck_flash_mtp_cache_restore and not args.recheck_flash_mtp_cache_admission:
        parser.error("--recheck-flash-mtp-cache-restore requires --recheck-flash-mtp-cache-admission")
    if args.borrow_27b_checkpoint_row and profile.id != 'qwen3.8-27b':
        parser.error("--borrow-27b-checkpoint-row requires the 27B model")
    if args.credit_27b_live_warm_cache and profile.id != "qwen3.8-27b":
        parser.error("--credit-27b-live-warm-cache requires the 27B model")
    if args.recheck_27b_cache_admission and profile.id != "qwen3.8-27b":
        parser.error("--recheck-27b-cache-admission requires the 27B model")
    if args.recheck_27b_first_store and not args.recheck_27b_cache_admission:
        parser.error("--recheck-27b-first-store requires --recheck-27b-cache-admission")
    if args.draft and not Path(args.draft).is_dir():
        parser.error("draft must be an existing local directory")
    if args.expanded_prefill and not profile.dense_prefill_expansion:
        parser.error("expanded-prefill is only validated for the dense 27B model")
    if args.draft:
        import json
        draft_config = json.loads((Path(args.draft) / 'config.json').read_text())
        if draft_config.get('model_type') != profile.draft_type:
            parser.error(f"this target requires draft model_type={profile.draft_type}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Upstream's 600-second default cancels valid 200K cold prefills before
    # they produce a token. Keep a finite, explicit service deadline.
    os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] = str(args.token_timeout)
    # Cold comparisons default to no APC. Disk persistence requires explicit opt-in.
    os.environ["APC_ENABLED"] = "1" if args.cache_mb else "0"
    os.environ.update(disk_environment)
    if args.cache_mb:
        os.environ["APC_MEMORY_MAX_GB"] = str(args.cache_mb * 1_000_000 / (1 << 30))
        os.environ["APC_HASH"] = "sha256"
    import mlx.core as mx
    mx.set_memory_limit(args.memory_mb * 1_000_000)
    mx.set_cache_limit(args.allocator_cache_mb * 1_000_000)
    from mlx_vlm.server.cli import main as serve
    if args.fused_prefill:
        from .fused_attention import install_fused_prefill
        install_fused_prefill()
    if args.credit_27b_live_warm_cache:
        from .warm_reserve import install_27b_warm_reserve
        install_27b_warm_reserve()
    if args.credit_live_warm_cache or args.credit_flash_mtp_live_warm_cache:
        from .warm_reserve import install_warm_reserve
        install_warm_reserve(allow_mtp=args.credit_flash_mtp_live_warm_cache)
    if args.borrow_single_checkpoint_row or args.borrow_flash_mtp_checkpoint_row:
        from .snapshot_views import install_snapshot_views
        install_snapshot_views()
    if args.borrow_27b_checkpoint_row:
        from .snapshot_views import install_27b_snapshot_views
        install_27b_snapshot_views()
    if args.recheck_27b_cache_admission:
        from .admission_recheck import install_27b_admission_recheck
        install_27b_admission_recheck(retry_empty_store=args.recheck_27b_first_store)
    if args.recheck_flash_mtp_cache_admission:
        from .admission_recheck import install_flash_admission_recheck
        install_flash_admission_recheck()
    if args.recheck_flash_mtp_cache_restore:
        from .restore_admission import install_flash_restore_recheck
        install_flash_restore_recheck()
    if args.normalize_single_warm_cache:
        from .warm_cache import install_single_warm_cache
        install_single_warm_cache()
    if args.release_finished_mtp:
        if not args.draft:
            parser.error("--release-finished-mtp requires --draft")
        from .mtp_lifecycle import install_mtp_cleanup
        install_mtp_cleanup()
    if args.expanded_prefill:
        import logging
        from mlx_vlm.server.generation import ResponseGenerator
        from .prefill_weights import expand_prefill_weights
        initialize = ResponseGenerator._initialize_model

        def initialize_with_expanded_prefill(self):
            # Pinned backend hook: run on its generation thread before readiness.
            initialize(self)
            report = expand_prefill_weights(self.model.language_model,
                                             args.memory_mb * 1_000_000)
            logging.getLogger(__name__).info("Expanded prefill weights: %s", report)

        ResponseGenerator._initialize_model = initialize_with_expanded_prefill
    sys.argv = ["mlx_vlm.server", "--model", str(path), "--host", "127.0.0.1",
                "--port", str(args.port), "--max-num-seqs", "1",
                "--max-kv-size", str(args.context), "--prefill-step-size", "2048"]
    if args.draft:
        sys.argv.extend(["--draft-model", str(Path(args.draft).resolve()),
                         "--draft-kind", "mtp", "--draft-block-size", str(args.block)])
    serve()


if __name__ == "__main__":
    main()
