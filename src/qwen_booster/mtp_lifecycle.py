"""Release completed speculative batch state for the pinned MLX-VLM backend.

The backend retains the empty batch until the next prompt finishes. Its suspended
round generator also closes over the batch, keeping a full target cache alive.
APC owns independent snapshots; these request-local references are no longer needed.
"""


def release_finished_batch(batch):
    if len(batch):
        return False
    fields = ('hidden', 'shared_kv_states', 'prompt_tokens', 'first_tokens')
    rounds = batch._rounds_iter
    if rounds is None and not batch.prompt_cache and all(getattr(batch, f) is None for f in fields):
        return False
    # Close while its inputs are still available for generator finalizers.
    if rounds is not None:
        rounds.close()
    batch._rounds_iter = None
    batch.prompt_cache = []
    for name in fields:
        setattr(batch, name, None)
    return True


def patch_batch_class(batch_class, after_release=None):
    if getattr(batch_class, '_qwen_booster_cleanup_installed', False):
        return

    def wrap(method):
        def call(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            if release_finished_batch(self) and after_release is not None:
                after_release()
            return result
        return call

    batch_class.next = wrap(batch_class.next)
    batch_class.filter = wrap(batch_class.filter)
    batch_class._qwen_booster_cleanup_installed = True


def install_mtp_cleanup():
    import gc
    import logging
    import mlx.core as mx
    from mlx_vlm.generate.ar import SpeculativeGenerationBatch

    def collected():
        gc.collect()
        mx.clear_cache()
        logging.getLogger(__name__).info(
            'Released finished MTP batch: active_bytes=%d allocator_bytes=%d',
            mx.get_active_memory(), mx.get_cache_memory())

    patch_batch_class(SpeculativeGenerationBatch, collected)
