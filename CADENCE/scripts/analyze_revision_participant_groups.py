#!/usr/bin/env python
"""Aggregate known repeated development visits before revision group inference."""
import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_revision_statistics import read_csv,write_csv,group_tests,METRICS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_dir',default='CADENCE/revision_tracking/analysis/participant_grouped')
    parser.add_argument('--record_metrics', default='CADENCE/revision_tracking/analysis/reviewer_statistics/patient_affective_metrics.csv')
    parser.add_argument('--source_metadata', required=True)
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
    source=Path(args.record_metrics)
    metadata=Path(args.source_metadata)
    records=[r for r in read_csv(source) if r['split']=='development']
    metas={r['adressfname']:r for r in read_csv(metadata) if r['test']=='FALSE'}
    assert {r['conv_id'] for r in records}==set(metas)
    grouped=defaultdict(list)
    for r in records:
        m=metas[r['conv_id']]
        # Pitt sid identifies longitudinal participant; do not apply its semantics to other corpora.
        key=m['dataset']+':'+(m['sid'] if m['dataset']=='Pitt' else m['id'])
        grouped[key].append(r)
    features=METRICS+[k for k in records[0] if k.startswith('mean_p_')]
    units=[];mapping=[]
    for key,visits in sorted(grouped.items()):
        diagnoses={r['diagnosis'] for r in visits};assert len(diagnoses)==1
        row=dict(conv_id=key,diagnosis=next(iter(diagnoses)),n_recordings=len(visits),
            recording_ids=';'.join(sorted(r['conv_id'] for r in visits)),
            identity_basis='Pitt sid' if key.startswith('Pitt:') else 'source recording proxy; relies on corpus construction')
        for k in features:
            vals=np.array([float(r[k]) for r in visits]);assert np.isfinite(vals).all();row[k]=float(vals.mean())
        units.append(row)
        for r in visits:mapping.append(dict(recording_id=r['conv_id'],participant_group=key,diagnosis=r['diagnosis'],group_n_recordings=len(visits)))
    assert len(units)==162 and sum(r['diagnosis']=='cn' for r in units)==75 and sum(r['diagnosis']=='ad' for r in units)==87
    stats=group_tests(units,np.random.default_rng(43),10000,10000)
    out.mkdir(parents=True)
    write_csv(out/'participant_affective_metrics.csv',units);write_csv(out/'recording_to_participant.csv',mapping)
    write_csv(out/'affective_group_tests.csv',stats)
    protocol=dict(date='2026-10-12',unit='Pitt participants; other corpora source-record proxies',n_records=166,n_units=162,n_ad=87,n_cn=75,
        aggregation='arithmetic mean of per-record metrics within each known participant; equal weight for visits then equal weight for participants; no concatenation across visits',
        affective_features=features,seed=43,permutations=10000,bootstrap_repeats=10000,
        multiplicity='Holm for five affective metrics',ci='pointwise 95% percentile participant bootstrap within diagnosis',effect='AD-CN mean difference and Hedges g',
        analysis_status='revision analysis, not preregistered',identity_limit='16 non-Pitt development records use distinct source-record IDs; person identities not independently verified',
        replaces='reviewer_statistics/affective_group_tests.csv for inferential use',training_performed=False,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,metadata]})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    labels=[k[len('mean_p_'):] for k in features if k.startswith('mean_p_')]
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(labels))
    for diag,offset,color in [('cn',-.18,'#4477AA'),('ad',.18,'#CC6677')]:
        matrix=np.array([[r['mean_p_'+k] for k in labels] for r in units if r['diagnosis']==diag])
        ax.bar(x+offset,matrix.mean(axis=0),.36,yerr=matrix.std(axis=0,ddof=1)/np.sqrt(len(matrix)),capsize=3,label=f'{diag.upper()} (n={len(matrix)})',color=color)
    ax.set_xticks(x,labels,rotation=25);ax.set_ylabel('Mean predicted emotion probability');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'emotion_distribution_grouped.{ext}',dpi=250)
    plt.close(fig)
    ylabel=dict(neutral='Mean neutral probability',entropy='Mean entropy (nats)',volatility='Mean consecutive embedding L2 distance',switch_rate='Dominant emotion switching rate',dwell='Mean run length (segments)')
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(4.5,4))
        ax.boxplot([[r[metric] for r in units if r['diagnosis']==d] for d in ['cn','ad']],tick_labels=['CN','AD'],showmeans=True)
        r=next(r for r in stats if r['metric']==metric)
        ax.set(ylabel=ylabel[metric],title=f'Grouped development; Holm p={r["p_holm"]:.3g}');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'{metric}_grouped.{ext}',dpi=250)
        plt.close(fig)
    lines=['# Kişi gruplarına göre duygu analizi — 12 Ekim 2026','','166 development kaydı, bilinen Pitt ziyaretleri birleştirilince 162 analiz birimi: 87 AD / 75 CN. 150 Pitt development kaydı 146 kişiye indirgenir; diğer 16 kayıt corpus + özgün kayıt ID ile temsil edilir, bağımsız kişi kimlikleri ayrıca doğrulanmadı.','',
        'Her ziyaretin neutral/entropy/volatility/switch/dwell ölçümü önce kayıt içinde hesaplandı; aynı kişinin ziyaret ölçümleri aritmetik ortalamayla birleştirildi. Ziyaretler arası yapay geçiş oluşturulmadı. Her analiz birimi eşit ağırlıklı. Test kayıtları grup testlerine katılmadı.','',
        'Beş ölçüm için 10.000 permutation, tanı içinde 10.000 bootstrap, %95 noktasal CI ve Holm düzeltmesi. Grup SD ddof=1. Sonuçlar otomatik encoder çıktılarının düzeltilmemiş grup ilişkileridir; klinik duygu ölçümü değildir.','',
        '| Ölçüm | AD−CN | %95 CI | Holm p |','|---|---:|---|---:|']
    for r in stats:lines.append(f'| {r["metric"]} | {r["difference_ad_minus_cn"]:.4f} | [{r["ci_low"]:.4f}, {r["ci_high"]:.4f}] | {r["p_holm"]:.6g} |')
    lines+=['','Eski kayıt düzeyindeki CSV/figürler silinmedi; makaledeki beş duygu testi için bu paket kullanılmalı. Model/CV/test tahminleri değiştirilmedi. Resmî testin anonim kişi eşlemesi çözülemediğinden test sonuçları kişi bağımsız genelleme olarak sunulmamalı.','',
        'Yeniden üretim: `python CADENCE/scripts/analyze_revision_participant_groups.py --output_dir YENI_BOS_KLASOR`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    for r in stats:print(r['metric'],r['ad_n'],r['cn_n'],r['difference_ad_minus_cn'],r['p_holm'])

if __name__=='__main__':main()
