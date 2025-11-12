from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

def _load_json(p: Optional[str]) -> Optional[Dict]:
    if not p:
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {p}: {e}")
        return None

def _extract_phoneme_sequence(config: Optional[Dict]) -> List[Dict]:
    """Return [{'arpabet_phoneme': str, 'display_time': float, 'ui_latency_ms': float?}, ...]."""
    if not config:
        return []
    seq = config.get('phoneme_sequence')
    if isinstance(seq, list) and seq:
        out = []
        for x in seq:
            arp = x.get('arpabet_phoneme') or x.get('arpabet')
            t = x.get('display_time') or x.get('recognition_time')
            if arp is None or t is None:
                continue
            item = {'arpabet_phoneme': str(arp), 'display_time': float(t)}
            if 'ui_latency_ms' in x:
                item['ui_latency_ms'] = float(x['ui_latency_ms'])
            out.append(item)
        if out:
            return out

    # Fallback reconstruction (older results): use log_data or results
    items = []
    if isinstance(config.get('log_data'), list):
        items = config['log_data']
    elif isinstance(config.get('results'), list):
        items = config['results']
    if not items:
        return []

    add_sec = 0.0
    if isinstance(config.get('target_latency_ms'), (int, float)):
        add_sec = float(config['target_latency_ms']) / 1000.0

    out = []
    for ev in items:
        arp = ev.get('arpabet_phoneme') or ev.get('arpabet')
        if not arp:
            continue
        t = ev.get('display_time', ev.get('recognition_time'))
        if t is None:
            continue
        out.append({'arpabet_phoneme': str(arp), 'display_time': float(t) + add_sec})
    return out


def analyze_phoneme_accuracy(config1_data: Dict, config_data: Dict) -> Optional[Dict]:
    base = _extract_phoneme_sequence(config1_data)
    test = _extract_phoneme_sequence(config_data)
    if not base or not test:
        return None
    b = [p['arpabet_phoneme'] for p in base]
    t = [p['arpabet_phoneme'] for p in test]
    L = min(len(b), len(t))
    if L == 0:
        return None
    matches = sum(1 for i in range(L) if b[i] == t[i])
    acc = (matches / len(b)) * 100.0 if b else 0.0
    return {
        'accuracy': float(acc),
        'baseline_count': int(len(b)),
        'test_count': int(len(t)),
        'matches': int(matches),
        'missed': max(0, len(b) - len(t)),
        'extra': max(0, len(t) - len(b)),
    }


def calculate_timing_deviation(config1_data: Dict, config_data: Dict) -> Optional[Dict]:
    base = _extract_phoneme_sequence(config1_data)
    test = _extract_phoneme_sequence(config_data)
    if not base or not test:
        return None
    bt = [p['display_time'] for p in base]
    tt = [p['display_time'] for p in test]
    L = min(len(bt), len(tt))
    if L == 0:
        return None
    dev_ms = [abs(tt[i] - bt[i]) * 1000.0 for i in range(L)]
    return {
        'mean_deviation_ms': float(np.mean(dev_ms)),
        'std_deviation_ms': float(np.std(dev_ms)),
        'max_deviation_ms': float(np.max(dev_ms)),
        'min_deviation_ms': float(np.min(dev_ms)),
    }


def summarize_ui_latency(config_data: Dict) -> Optional[Dict]:
    seq = _extract_phoneme_sequence(config_data)
    vals = [p['ui_latency_ms'] for p in seq if 'ui_latency_ms' in p]
    if not vals:
        return None
    return {
        'ui_latency_mean_ms': float(np.mean(vals)),
        'ui_latency_std_ms': float(np.std(vals)),
        'ui_latency_min_ms': float(np.min(vals)),
        'ui_latency_max_ms': float(np.max(vals)),
    }


def format_report(config1_for_impl2: Optional[Dict], config2: Optional[Dict],
                  config1_for_impl3: Optional[Dict], config3: Optional[Dict],
                  accuracy2: Optional[Dict], accuracy3: Optional[Dict],
                  timing2: Optional[Dict], timing3: Optional[Dict],
                  ui2: Optional[Dict], ui3: Optional[Dict]) -> str:
    lines: List[str] = []
    lines.append("COMPARATIVE ANALYSIS REPORT")
    lines.append("===========================\n")

    def _block(title: str, cfg: Optional[Dict], acc: Optional[Dict], tim: Optional[Dict], ui: Optional[Dict]):
        lines.append(title)
        lines.append("-" * len(title))
        if not cfg:
            lines.append("No data.\n")
            return
        lines.append(f"avg_model_latency_ms: {cfg.get('avg_inference_latency_ms', 0):.2f}")
        if 'jitter_coefficient' in cfg:
            lines.append(f"jitter_coeff(%): {cfg.get('jitter_coefficient', 0):.2f}")
        if 'min_interval_ms' in cfg and 'max_interval_ms' in cfg:
            lines.append(f"interval_range_ms: {cfg.get('min_interval_ms', 0):.2f} - {cfg.get('max_interval_ms', 0):.2f}")
        if ui:
            lines.append(
                f"ui_latency_ms: mean {ui['ui_latency_mean_ms']:.2f}, std {ui['ui_latency_std_ms']:.2f}, "
                f"min {ui['ui_latency_min_ms']:.2f}, max {ui['ui_latency_max_ms']:.2f}"
            )
        if acc:
            lines.append(f"accuracy_vs_baseline(%): {acc['accuracy']:.2f}")
            lines.append(f"baseline_count: {acc['baseline_count']}  test_count: {acc['test_count']}")
            lines.append(f"matches: {acc['matches']}  missed: {acc['missed']}  extra: {acc['extra']}")
        else:
            lines.append("accuracy: N/A")
        if tim:
            lines.append(
                f"timing_deviation_ms: mean {tim['mean_deviation_ms']:.2f}, std {tim['std_deviation_ms']:.2f}, "
                f"min {tim['min_deviation_ms']:.2f}, max {tim['max_deviation_ms']:.2f}"
            )
        else:
            lines.append("timing deviation: N/A")
        lines.append("")

    _block("Config 2 (Impl-2)", config2, accuracy2, timing2, ui2)
    _block("Config 3 (Impl-3)", config3, accuracy3, timing3, ui3)

    return "\n".join(lines)


def generate_comparative_analysis(config1_for_impl2_path: str,
                                  config2_path: str,
                                  config1_for_impl3_path: str,
                                  config3_path: str,
                                  out_text_path: Optional[str] = None) -> Optional[Dict]:
    c1_2 = _load_json(config1_for_impl2_path)
    c2 = _load_json(config2_path)
    c1_3 = _load_json(config1_for_impl3_path)
    c3 = _load_json(config3_path)

    accuracy2 = analyze_phoneme_accuracy(c1_2, c2) if c1_2 and c2 else None
    accuracy3 = analyze_phoneme_accuracy(c1_3, c3) if c1_3 and c3 else None
    timing2 = calculate_timing_deviation(c1_2, c2) if c1_2 and c2 else None
    timing3 = calculate_timing_deviation(c1_3, c3) if c1_3 and c3 else None
    ui2 = summarize_ui_latency(c2) if c2 else None
    ui3 = summarize_ui_latency(c3) if c3 else None

    report = format_report(c1_2, c2, c1_3, c3, accuracy2, accuracy3, timing2, timing3, ui2, ui3)

    out = {
        'accuracy2': accuracy2,
        'accuracy3': accuracy3,
        'timing2': timing2,
        'timing3': timing3,
        'ui2': ui2,
        'ui3': ui3,
        'report': report,
    }

    if out_text_path:
        try:
            p = Path(out_text_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(report, encoding='utf-8')
            logging.info(f"Analysis report saved to {p}")
        except Exception as e:
            logging.error(f"Failed to save analysis text: {e}")

    return out