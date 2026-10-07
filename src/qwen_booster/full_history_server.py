"""Experimental launcher preserving supplied reasoning history in Qwen requests."""


def configure_environment(argv=None):
    """Set import-time backend defaults before installing the history adapter."""
    import argparse
    import math
    import os
    from .vlm_server import disk_cache_environment

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cache-mb', type=int, default=0)
    parser.add_argument('--disk-cache-dir')
    parser.add_argument('--disk-cache-mb', type=int, default=0)
    parser.add_argument('--token-timeout', type=float, default=7200)
    args, _ = parser.parse_known_args(argv)
    if args.cache_mb < 0 or not math.isfinite(args.token_timeout) or args.token_timeout <= 0:
        parser.error('invalid cache budget or token timeout')
    try:
        disk = disk_cache_environment(args.disk_cache_dir, args.disk_cache_mb, args.cache_mb)
    except ValueError as error:
        parser.error(str(error))
    os.environ.update(disk)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      MLX_VLM_TOKEN_QUEUE_TIMEOUT=str(args.token_timeout),
                      APC_ENABLED='1' if args.cache_mb else '0')
    if args.cache_mb:
        os.environ.update(APC_MEMORY_MAX_GB=str(args.cache_mb * 1_000_000 / (1 << 30)),
                          APC_HASH='sha256')


def main():
    configure_environment()
    from .history_adapter import install
    install()
    from .vlm_server import main as serve
    serve()


if __name__ == '__main__':
    main()
