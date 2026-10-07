"""Index reviewed workload evidence without inventing an overall winner.

This aggregates existing reviews, not a replacement for independent grading.
Retains limitations and actual cache regimes alongside measured latencies.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sources = {}

    def read(name):
        path = Path('results') / name
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        review = json.loads(raw)
        # Where reviews record raw evidence hashes, reject changed evidence.
        hashes = review.get('source_sha256', {})
        hashes = hashes if isinstance(hashes, dict) else {}
        for name, expected in hashes.items():
            evidence = Path(name)
            if hashlib.sha256(evidence.read_bytes()).hexdigest() != expected:
                raise ValueError('Changed reviewed evidence: ' + name)
        return review

    short = read('multi-model-short-quality-comparison-v1.json')
    code = read('multi-model-200k-code-comparison-v1.json')
    repository = read('multi-model-200k-repository-comparison-v1.json')
    conversation = read('flash-next-snapshot-views-conversation-review-v1.json')
    mtp_conversation = read('mtp-200k-conversation-v3-strict-review.json')
    mtp_raw = Path('results/mtp-200k-conversation-v3.jsonl')
    if hashlib.sha256(mtp_raw.read_bytes()).hexdigest() != mtp_conversation['source_sha256']:
        raise ValueError('Changed reviewed MTP conversation evidence')
    repeated_code = read('flash-next-snapshot-views-long-code-review-v1.json')
    boundary = read('flash-next-native-boundary-review-v1.json')
    disk = read('flash-next-disk-200k-v1-review.json')
    native_failure = read('native-27b-failure-replay-review-v1.json')
    branches = read('flash-next-followup-branches-200k-review-v1.json')
    native_branches = read('native-27b-followup-branches-200k-review-v1.json')
    prefill = read('query-sliced-prefill-full-model-review-v1.json')
    normalization_short = read('27b-warm-normalize-screen-review-v1.json')
    normalization_200k = read('27b-warm-normalize-200k-review-v1.json')
    normalization_interference = read('27b-warm-normalize-conversation-interference-review-v1.json')
    normalization_conversation = read('27b-warm-normalize-conversation-review-v2.json')
    interleaving_short = read('27b-cache-interleaving-short-review-v1.json')
    interleaving_200k = read('27b-cache-interleaving-200k-review-v1.json')
    normalization_repeated = read('27b-warm-normalize-repeated-v1-review.json')
    vlm_failure = read('vlm-27b-261k-replay-v1-review.json')
    native_q8 = read('native-27b-8bit-261k-review-v1.json')
    diverse_reasoning = read('diverse-reasoning-v1-review.json')
    attention_progress = read('diverse-reasoning-v1-prefill-progress-analysis.json')
    head_group_kernel = read('head-group-attention-screen-v1-review.json')
    head_group_model = read('head-group-full-model-screen-v1-review.json')
    head_group_long = read('head-group-full-model-200k-v1-review.json')
    head_group_repeated = read('head-group-full-model-200k-repeated-v1-review.json')
    prefill_windows = read('prefill-windows-200k-v1-review.json')
    fused_short = read('fused-distributions-short-v1-reproducible-review.json')
    fused_long = read('fused-distributions-200k-v1-review.json')
    fused_tasks_short = read('fused-tasks-short-v1-review.json')
    fused_tasks_long = read('fused-tasks-200k-v1-review.json')
    fused_tasks_repeated = read('fused-tasks-200k-repeated-v1-review.json')
    fused_tasks_repeated_stats = read('fused-tasks-200k-repeated-v1-summary.json')
    fused_http_mtp = read('fused-http-mtp-v1-review.json')
    fused_conversation = read('fused-mtp-conversation-200k-v1-review.json')
    combined_cache = {
        'native_original_partial': read('fused-mtp-27b-261k-v1-partial-review.json'),
        'native_admission_failure': read('fused-mtp-27b-261k-admission-v1-review.json'),
        'borrow_short': read('borrow-27b-http-short-v1-review.json'),
        'borrow_native_failure': read('borrow-27b-261k-admission-v1-review.json'),
        'credit_short': read('credit-27b-http-short-v1-review.json'),
        'credit_native_cache_pass_quality_failure': read('credit-27b-261k-admission-v1-review.json'),
        'credit_conversation_200k': read('credit-27b-conversation-200k-v1-review.json'),
        'credit_switching_200k': read('credit-27b-switching-200k-v1-review.json'),
        'credit_switching_admission_diagnostic': read('credit-27b-switching-admission-200k-v1-review.json'),
        'credit_switching_ten_rounds': read('credit-27b-switching-conversation-200k-v1-review.json'),
        'recheck_short': read('recheck-27b-http-short-v2-review.json'),
        'recheck_switching_ten_rounds': read('recheck-27b-switching-conversation-200k-v1-review.json'),
        'combined_repository_cache_failure': read('combined-27b-repository-v1-review.json'),
        'first_store_short': read('first-store-27b-http-short-v1-review.json'),
        'first_store_repository_diagnostic': read('first-store-27b-repository-replay-v1-review.json'),
    }
    switching_27b = read('document-switching-200k-v1-27b-partial-review.json')
    switching_flash = read('document-switching-200k-flash-v1-partial-review.json')
    switching_final = read('document-switching-final-only-200k-v1-flash-partial-review.json')
    switching_admission = read('document-switching-admission-200k-v1-review.json')
    switching_disk = read('document-switching-disk-200k-v1-review.json')
    switching_27b_complete = read('document-switching-disk-27b-200k-v1-review.json')
    switching_27b_restart = read('document-switching-27b-restart-v1-review.json')
    repository_behavior = read('repository-behavior-inference-v1-review.json')
    code_rows = []
    for model, pair in code['cases'].items():
        code_rows.append({
            'configuration': model,
            'cold_correct': pair['cold']['independent_grade'],
            'warm_correct': pair['warm']['independent_grade'],
            'body_and_reasoning_parity': pair['answer_parity'] and pair['reasoning_parity'],
            'observations': {regime: {
                'seconds': pair[regime]['seconds'],
                'first_answer_seconds': pair[regime]['first_visible_answer_seconds'],
                'input_tokens': pair[regime]['usage']['prompt_tokens'],
                'output_tokens': pair[regime]['usage']['completion_tokens'],
                'cached_tokens': pair[regime]['server_timings']['cache_n'],
                'cold_prefill_tokens_per_second': (pair[regime]['server_timings']['prompt_per_second']
                                                   if regime == 'cold' else None),
                'decode_tokens_per_second': pair[regime]['server_timings']['predicted_per_second'],
                'server_prompt_seconds': pair[regime]['server_timings']['prompt_ms'] / 1000,
                'server_generation_seconds': pair[regime]['server_timings']['predicted_ms'] / 1000,
                'peak_mlx_gb': pair[regime]['server_timings']['peak_memory'],
            } for regime in ('cold', 'warm')}})
    repository_rows = {}
    for model, cases in repository['models'].items():
        repository_rows[model] = {
            'correct_configurations': sum(c['independent_grade'] and c['answer_complete'] for c in cases),
            'configuration_count': len(cases),
            'cases': [{
                'id': c['case_id'], 'correct': c['independent_grade'],
                'answer_complete': c['answer_complete'], 'finish_reason': c['finish_reason'],
                'seconds': c['seconds'], 'cached_tokens': c['server_timings']['cache_n'],
                'input_tokens': c['usage']['prompt_tokens'],
                'output_tokens': c['usage']['completion_tokens'],
            } for c in cases]}
    report = {
        'status': 'in_progress', 'acceptance_complete': False,
        'scope': __doc__, 'source_review_sha256': sources,
        'short_tasks': {
            'case_count': len(short['cases']),
            'models': {model: {k: result[k] for k in ('correct', 'thinking_correct', 'nonthinking_correct')}
                       for model, result in short['models'].items()},
            'limit': 'Four task types with thinking on/off; one observation each. Different cache/context settings; no controlled cross-model speedup.'},
        'cross_file_200k': {
            'observations': code_rows, 'limit': code['scope'],
            'rate_definition': 'Server-reported prompt and generated tokens per second. '
                               'Decode includes reasoning and answer tokens. Warm requests reuse all but '
                               'one input token, so warm prompt throughput is not a full-input prefill rate. '
                               'Single paired observations, not repeated stable performance estimates.'},
        'repository_200k': {
            'models': repository_rows,
            'limit': 'One repository question with four generation settings, not four independent tasks. Actual cache regimes differ.'},
        'diverse_reasoning_short_and_200k': diverse_reasoning,
        'repository_behavior_200k': repository_behavior,
        'two_distinct_documents_200k': {
            'required_daily_workload': True,
            '27b_original_partial': switching_27b,
            'flash_original_partial': switching_flash,
            'flash_final_checkpoint_only_partial': switching_final,
            'flash_admission_diagnostic': switching_admission,
            'flash_ram_ssd_complete': switching_disk,
            '27b_ram_switching_complete_ssd_unproven': switching_27b_complete,
            '27b_separate_fresh_process_ssd_recovery': switching_27b_restart,
            'accepted': False,
            'limit': 'Original configurations lose long-document cache. Final-only Flash retains A '
                     'after A completes but evicts it during B; revisit starts full recompute. '
                     'Flash RAM+SSD subsequently passes 8 responses and 4 full-history/output pairs, '
                     'with 2 actual disk restores and fresh disk-disabled cold controls. '
                     '27B final-only with RAM+SSD configured passes 8 responses and 4 pairs, all warm restores from RAM; a separate two-fixture fresh-process screen verifies SSD recovery. Broad daily-use acceptance remains open.'},
        'flash_candidate_200k': {
            'ten_turn_and_final_cold_gate': conversation['bounded_ten_turn_check_passed'],
            'warm_first_answer_median_seconds': conversation['observed_warm_first_answer_median_seconds'],
            'three_repeated_long_code_gate': repeated_code['bounded_long_code_gate_passed'],
            'default_enabled': False,
            'limit': 'Synthetic retrieval and one code task; no P95 or broad daily-use quality claim.'},
        '27b_mtp_conversation_200k': {
            'ten_turn_and_final_cold_gate': mtp_conversation['bounded_ten_turn_check_passed'],
            'observed_turns': mtp_conversation['observed_turns'],
            'warm_first_answer_median_seconds': mtp_conversation['observed_warm_first_answer_median_seconds'],
            'actual_input_range': [min(t['input_tokens'] for t in mtp_conversation['turns']),
                                   max(t['input_tokens'] for t in mtp_conversation['turns'])],
            'limit': mtp_conversation['scope'] + ' Background model download overlapped the first cold prefill; not a controlled timing comparison.'},
        'flash_native_boundary': {
            'capacity_quality_gate': boundary['bounded_native_capacity_quality_passed'],
            'correct_requests': boundary['correct_requests'],
            'cached_pairs_passed': boundary['cache_reuse_pairs_passed'],
            'paired_cases': len(boundary['checks']), 'observations': boundary['checks'],
            'limit': boundary['scope']},
        'flash_disk_200k': disk,
        '27b_native_failure_cold_replay': native_failure,
        '27b_vlm_261k_failure_replay': vlm_failure,
        '27b_native_q8_261k_diagnostic': native_q8,
        'flash_followup_branches_200k': branches,
        '27b_native_followup_branches_200k': native_branches,
        '27b_query_sliced_prefill_screen': prefill,
        '27b_fused_attention_numerical_diagnostics': {
            'short': fused_short, '200k': fused_long,
            'default_enabled': False, 'acceptance_complete': False,
            'limit': 'Fixed baseline continuations only; 96 short and 32 long positions '
                     'matched top tokens. Free-generation quality and speed remain unverified.'},
        '27b_fused_attention_short_task_screen': fused_tasks_short,
        '27b_fused_attention_200k_task_screen': fused_tasks_long,
        '27b_fused_attention_200k_repeated': {'review': fused_tasks_repeated, 'statistics': fused_tasks_repeated_stats},
        '27b_fused_attention_http_mtp': fused_http_mtp,
        '27b_fused_mtp_conversation_200k': fused_conversation,
        '27b_combined_cache_candidates': {
            'reviews': combined_cache,
            'default_enabled': False, 'acceptance_complete': False,
            'limit': 'Ten-turn 200K synthetic retrieval passes; two-document timing run '
                     'has eight correct answers but skips one followup snapshot. A separate '
                     'traced run passes both exact extended-history revisits without a skipped '
                     'store. The subsequent 28-request alternating ten-round run reproduces three skipped stores: '
                     '28 answers and four cold/warm pairs pass, but only 21 of 24 latest-prefix checks pass. '
                     'The bounded post-eviction recheck candidate then passes 28 answers, all 24 latest-prefix checks, and four pairs; three real rejected admissions recover within unchanged guards and no stores are skipped. General retention reliability remains unproven. Tracing '
                     'is not a speed benchmark. Native 261265-token cache retention passes '
                     'three repeats while all four answers remain wrong. Historical timing '
                     'differences do not establish isolated optimization speedups.'},
        '27b_head_group_attention_screen': {
            'full_prefix_window_diagnostics': prefill_windows,
            'prefill_progress_analysis': attention_progress,
            'isolated_attention': head_group_kernel,
            'full_model_8k_32k': head_group_model,
            'full_model_200k': head_group_long,
            'full_model_200k_three_pairs': head_group_repeated,
            'default_enabled': False,
            'limit': 'Memory reduction observed; 8K/32K formal timing pairs were slower. '
                     'One gated 200K pair preserved 64 greedy tokens and reduced TTFT by 4.06% '
                     'and peak MLX memory by 28.59%, with lower observed decode throughput. '
                     'The subsequent three AB/BA/AB pairs preserved all 64 greedy tokens per pair, '
                     'but median paired TTFT speedup was 0.9996091170913836 (no median gain). '
                     'Peak MLX memory fell from 52.31GB to 37.36–39.63GB. '
                     'No stable speedup or semantic-quality acceptance.'},
        '27b_warm_normalization': {
            'short_code': normalization_short,
            'code_200k': normalization_200k,
            'conversation_interference': normalization_interference,
            'dedicated_port_conversation': normalization_conversation,
            'interleaving_short': interleaving_short,
            'interleaving_200k': interleaving_200k,
            'repeated_code_200k': normalization_repeated,
            'default_enabled': False,
            'limit': 'Short and 200K single code-fixture cold/warm pairs passed. First conversation run '
                     'lost the long prefix after two non-fixture requests and was stopped. Dedicated-port '
                     'ten-turn/final-cold rerun passed. Short two-slot retention failed, four-slot passed; four-slot 200K sequential interleaving passed. '
                     'Three fresh-server observations per mode on one 206849-token code task passed; '
                     'warm decode median 3.20 to 10.06 tok/s without MTP. No broad quality or P95 acceptance.'},
        'selection_status': 'No overall winner or automatic workload routing accepted yet.',
        'pending': [
            'Current-stack repository run stopped at 6/8 responses: content 6/6, bare JSON 2/6, one full warm cache miss. Diagnostic replay plus missing MTP control completed 4/4 content and two cache pairs, bare JSON 2/4. Both original admissions succeeded; zero first-store retry events, so recovery efficacy remains unproven. Keep first-store candidate default-off and original failure visible. Diagnostic timings and unequal generated outputs are not stable causal speedup evidence.',
            'Combined 27B fused/MTP/borrow/credit/recheck candidate passes the bounded 28-request alternating actual-history test with 24/24 latest-prefix hits and three actual guarded admission recoveries. Current-stack cross-file quality coverage, controlled repeat timing and broader pressure reliability remain pending; preserve earlier skip failures.',
            '27B Q4 261K cold replay reproduces the same adjacent-record error in native and VLM backends. Native Q8 answers this unchanged fixture correctly, but uses a different community conversion and different memory/cache budgets; isolated quantization causality is unproven. Broader long-context quality and correctly provisioned warm controls remain unresolved.',
            'Native-maximum disk fallback, disk branch/history recovery, edited-user branches and broader workload quality remain unverified. Both models passed the bounded 200K followup-branch fixture.',
            'Flash-Next MTP inference and same-model optimization comparison remain untested.',
            '27B singleton warm-cache optimization passed short/200K code pairs, three observations per mode, and dedicated-port ten-turn/final-cold conversation. Four-slot sequential 200K retention across two short requests passed; simultaneous concurrency, P95, broader workloads and combination with MTP remain unverified. Query-sliced prefill passed bounded 8K/32K token parity but showed no stable meaningful speed gain; remains disabled and unverified at 200K.',
        ],
    }
    with Path(args.output).open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({'output': args.output, 'review_sources': len(sources), 'acceptance_complete': False}))


if __name__ == '__main__':
    main()
