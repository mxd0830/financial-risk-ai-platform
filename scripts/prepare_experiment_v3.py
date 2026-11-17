"""离线复核旧结果，生成 500 人的配对评测清单；不调用模型。"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from build_missing_variant import FIELDS, rebuild_prompt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
OUT = DATA / 'experiment_v3'
SEED = 20261117
POLICY = (
    '\n缺失处理规则：全部字段必须保留；上述所有字段类型均允许 null。'
    '档案未提供的字段必须输出 JSON null，不得推测或从其他字段补算；'
    '不得用 0、空字符串或“未知”代替缺失值。明确提供的 0 必须保留为 0。'
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = DATA / 'eval_extraction.jsonl'
    legacy_source = DATA / 'eval_extraction_missing.jsonl'
    legacy_result = DATA / 'eval_missing_results.json'
    old = json.loads(legacy_result.read_text())
    old_n = old['summary']['n']
    old_rows = read_jsonl(legacy_source)[:old_n]
    old_ids = {r['id'] for r in old_rows}
    failures = old['failures']
    assert len({f['id'] for f in failures}) == len(failures)
    assert {f['id'] for f in failures} <= old_ids
    assert len(failures) == round(old_n * (1 - old['summary']['all_ok_rate']))

    categories = Counter()
    field_events = Counter()
    for f in failures:
        for key in f['fabricated']:
            value = f['parsed'][key]
            category = ('empty_string' if value == '' else
                        'numeric_zero' if type(value) in (int, float) and value == 0 else
                        'unknown_string' if value == '未知' else 'other_non_null')
            categories[category] += 1
            field_events[key] += 1
    assert sum(categories.values()) == old['summary']['fabricated_events']
    audit = {
        'source_sha256': digest(legacy_result),
        'original_summary': old['summary'],
        'missing_non_null_categories': dict(categories),
        'field_non_null_counts': dict(field_events),
        'limitations': [
            'Only failures have raw outputs; successful responses cannot be independently rescored.',
            'Old run has no saved ID manifest; first-N IDs inferred from script and checked against failures.',
            'Non-null missing fields are protocol violations, not necessarily fabricated factual claims.',
        ],
    }

    # 将已经用于错误分析的旧缺失样本排除，避免将开发案例放入新测试集。
    rows = [r for r in read_jsonl(source) if r['id'] not in old_ids]
    assert len({r['id'] for r in rows}) == len(rows)
    assert len(rows) >= 500
    rng = random.Random(SEED)
    rng.shuffle(rows)
    selected = rows[:500]
    cases = []
    for split, start, count, kind in [
        ('dev', 0, 100, 'missing'),
        ('test', 100, 300, 'missing'),
        ('test', 400, 100, 'complete'),
    ]:
        # 每 5 条覆盖全部 10 个字段，每条删除 2 个不同字段。
        pairs = []
        if kind == 'missing':
            for _ in range(count // 5):
                fields = FIELDS.copy()
                rng.shuffle(fields)
                pairs.extend([fields[i:i + 2] for i in range(0, len(fields), 2)])
        for index, row in enumerate(selected[start:start + count]):
            omitted = pairs[index] if kind == 'missing' else []
            gt = dict(row['ground_truth'])
            for field in omitted:
                gt[field] = None
            prompt = rebuild_prompt(gt, omitted) if omitted else row['prompt']
            conditions = ['A', 'B']
            if index % 2:
                conditions.reverse()
            cases.append({
                'id': row['id'], 'split': split, 'kind': kind,
                'omitted': omitted, 'ground_truth': gt,
                'prompts': {'A': prompt, 'B': prompt + POLICY},
                'condition_order': conditions,
            })
    assert len(cases) == len({r['id'] for r in cases}) == 500
    for split, expected in [('dev', 20), ('test', 60)]:
        counts = Counter(k for r in cases if r['split'] == split for k in r['omitted'])
        assert set(counts) == set(FIELDS) and set(counts.values()) == {expected}
    OUT.mkdir(exist_ok=True)
    for split in ['dev', 'test']:
        subset = [r for r in cases if r['split'] == split]
        rng.shuffle(subset)
        (OUT / f'{split}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset)
        )
    (OUT / 'legacy_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    config = {
        'status': 'prepared_not_run', 'sampling_seed': SEED,
        'unique_applicants': 500, 'dev_cases': 100, 'test_cases': 400,
        'planned_calls_one_pass': 1000,
        'dev_missing_fields_per_field': 20, 'test_missing_fields_per_field': 60,
        'excluded_legacy_missing_ids': sorted(old_ids),
        'source_sha256': digest(source),
        'dev_sha256': digest(OUT / 'dev.jsonl'), 'test_sha256': digest(OUT / 'test.jsonl'),
        'candidate_options': {'temperature': 0.1, 'num_ctx': 4096, 'num_predict': 256, 'seed': 42},
        'note': 'Output limit must be calibrated on dev data. Freeze model/backend and all parameters before test.',
    }
    (OUT / 'manifest.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'cases': len(cases), 'audit': dict(categories)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
