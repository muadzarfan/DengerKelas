"""Repeatable Step 3 driver; no model training or ASR benchmarking."""
import argparse
import time
from .common import ROOT,read_json,write_json,digest,event


def manifest_hashes():
    return {p.name:digest(p) for p in sorted((ROOT/'data/manifests').glob('*.jsonl'))
            if p.name.startswith(('fleurs_','demand_','train_e2_','validation_noisy','test_noisy'))}


def main(stage='all'):
    started=time.perf_counter()
    old=manifest_hashes()
    if stage in ('all','acquire'):
        from .acquire import acquire
        acquire()
    if stage in ('all','extract'):
        from .extract import extract_all
        extract_all()
    if stage in ('all','audit'):
        from .audit import audit_speech,audit_noise
        audit_speech(); audit_noise()
    if stage in ('all','recipes'):
        from .recipes import build_recipes
        build_recipes()
    if stage in ('all','verify'):
        from .verify import verify_all
        from .plots import plot_distributions
        verify_all(); plot_distributions()
    new=manifest_hashes()
    if stage=='all':
        result={'stage':stage,'elapsed_seconds':time.perf_counter()-started,
            'previous_manifests_present':bool(old),'previous_hashes':old,'current_hashes':new,
            'same_manifest_bytes':bool(old) and old==new,
            'status':'completed','meaning':'Identical JSONL hashes verify stable IDs, counts, rows, transforms and recipes; no duplicated append output'}
        write_json(ROOT/'results/step3/rerun_verification.json',result)
        event('preparation_complete',elapsed_seconds=result['elapsed_seconds'],same_manifest_bytes=result['same_manifest_bytes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','acquire','extract','audit','recipes','verify'],default='all')
    main(parser.parse_args().stage)
