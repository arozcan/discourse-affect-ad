#!/usr/bin/env python
"""Reproducible revision statistics from existing caches/predictions; no training."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ['neutral', 'entropy', 'volatility', 'switch_rate', 'dwell']
MODELS = ['dual_stream', 'text_only', 'emotion_only']
PAIRS = [(MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[2], MODELS[1])]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); result = np.empty(len(ps))
    result[order] = np.minimum(1, np.maximum.accumulate(ps[order] * np.arange(len(ps), 0, -1)))
    return result


def performance(y, p):
    tn = np.sum((y == 0) & (p == 0), axis=-1); tp = np.sum((y == 1) & (p == 1), axis=-1)
    fp = np.sum((y == 0) & (p == 1), axis=-1); fn = np.sum((y == 1) & (p == 0), axis=-1)
    n0, n1 = tn + fp, tp + fn
    f0 = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return (tn + tp) / (n0 + n1), (n0 * f0 + n1 * f1) / (n0 + n1)


def summary(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if not len(x): return dict(n=0, mean='', sd='', median='', q1='', q3='', min='', max='')
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x)>1 else '',
                median=float(np.median(x)), q1=float(np.quantile(x,.25)), q3=float(np.quantile(x,.75)),
                min=float(x.min()), max=float(x.max()))


def model_tests(predictions, labels, rng, repeats, permutations):
    rows = read_csv(predictions); groups = {}
    for r in rows: groups.setdefault((r['model_id'],r['fold']), []).append(r)
    assert set(groups) == {(m, f) for m in MODELS for f in ['1','2','3','4','5','ensemble']}
    for rs in groups.values():
        assert len(rs) == len(labels) and {r['conv_id'] for r in rs} == set(labels)
        for r in rs:
            prob = float(r['p_ad'])
            assert np.isfinite(prob) and 0 <= prob <= 1
            assert int(r['y_true']) == labels[r['conv_id']] and int(r['y_pred']) == int(prob >= .5)
    ids = sorted(labels); y = np.array([labels[i] for i in ids]); preds = {}
    for m in MODELS:
        rs = {r['conv_id']:r for r in groups[m,'ensemble']}
        maps = [{r['conv_id']:float(r['p_ad']) for r in groups[m,str(f)]} for f in range(1,6)]
        assert all(abs(float(rs[i]['p_ad'])-np.mean([d[i] for d in maps])) < 1e-10 for i in ids)
        preds[m] = np.array([int(rs[i]['y_pred']) for i in ids])
    boot = np.concatenate([rng.choice(np.flatnonzero(y==k), (repeats, sum(y==k)), replace=True) for k in [0,1]], axis=1)
    results = []
    for a,b in PAIRS:
        pa,pb = preds[a],preds[b]; ca,cb = pa==y,pb==y
        a_only,b_only = int(np.sum(ca & ~cb)),int(np.sum(cb & ~ca))
        obs = np.array(performance(y,pa))-np.array(performance(y,pb))
        ab = performance(y[boot],pa[boot]); bb = performance(y[boot],pb[boot])
        # Exact prediction-swap enumeration when at most 16 discordant predictions.
        discord = np.flatnonzero(pa!=pb); d=len(discord); exact=d<=16
        total=2**d if exact else permutations; extreme=0
        for start in range(0,total,512):
            size=min(512,total-start)
            swap=np.zeros((size,len(y)),dtype=bool)
            swap[:,discord]=((np.arange(start,start+size)[:,None] >> np.arange(d)) & 1).astype(bool) if exact else rng.integers(0,2,(size,d)).astype(bool)
            xa=np.where(swap,pb,pa); xb=np.where(swap,pa,pb)
            delta=performance(y,xa)[1]-performance(y,xb)[1]
            extreme+=int(np.sum(np.abs(delta)>=abs(obs[1])-1e-12))
        p_f1=extreme/total if exact else (extreme+1)/(total+1)
        p_acc=binomtest(a_only,a_only+b_only,.5).pvalue if a_only+b_only else 1.
        for j,(metric,p) in enumerate([('accuracy',p_acc),('weighted_f1',p_f1)]):
            lo,hi=np.quantile(ab[j]-bb[j],[.025,.975])
            results.append(dict(model_a=a,model_b=b,metric=metric,n=len(y),a_correct_b_wrong=a_only,
                a_wrong_b_correct=b_only,difference_a_minus_b=float(obs[j]),ci_low=float(lo),ci_high=float(hi),
                p_raw=float(p),test='exact McNemar' if j==0 else ('exact paired prediction swap' if exact else 'Monte Carlo paired prediction swap'),
                permutations=0 if j==0 else total))
    for metric in ['accuracy','weighted_f1']:
        selected=[r for r in results if r['metric']==metric]
        for r,p in zip(selected,holm([r['p_raw'] for r in selected])): r['p_holm']=float(p)
    return results


def load_cache(path, split):
    with Path(path).open('rb') as f: data=pickle.load(f)
    cs=data['conversations']; assert len({c['conv_id'] for c in cs})==len(cs)
    assert all(c['split']==split and c['diag_label'] in ['ad','cn'] for c in cs)
    neutral=data['emotion_labels'].index('neutral'); rows=[]
    for c in sorted(cs,key=lambda c:c['conv_id']):
        p=np.asarray(c['emo_probs'],dtype=float); h=np.asarray(c['emo_hidden'],dtype=float); n=len(c['speakers'])
        assert p.shape==(n,len(data['emotion_labels'])) and h.shape[0]==n and n>0
        assert np.isfinite(p).all() and np.isfinite(h).all() and (p>=0).all() and (p<=1).all()
        assert np.allclose(p.sum(axis=1),1,atol=1e-5)
        # Existing downstream 'all' speaker policy: no filtering, full-recording transcript.
        dominant=p.argmax(axis=1); switches=int(np.sum(dominant[1:]!=dominant[:-1]))
        row=dict(conv_id=c['conv_id'],split='development' if split=='train' else 'test',diagnosis=c['diag_label'],
            n_segments=n,n_used_segments=min(n,128),neutral=float(p[:,neutral].mean()),
            entropy=float(-np.sum(p*np.log(np.maximum(p,1e-12)),axis=1).mean()),
            volatility=float(np.linalg.norm(np.diff(h,axis=0),axis=1).mean()) if n>1 else float('nan'),
            switch_rate=switches/(n-1) if n>1 else float('nan'),dwell=n/(switches+1))
        row.update({f'mean_p_{label}':float(p[:,i].mean()) for i,label in enumerate(data['emotion_labels'])})
        rows.append(row)
    return rows


def group_tests(rows,rng,repeats,permutations):
    results=[]
    for metric in METRICS:
        ad=np.array([r[metric] for r in rows if r['diagnosis']=='ad']); ad=ad[np.isfinite(ad)]
        cn=np.array([r[metric] for r in rows if r['diagnosis']=='cn']); cn=cn[np.isfinite(cn)]
        observed=ad.mean()-cn.mean(); joined=np.r_[ad,cn]; extreme=0
        for _ in range(permutations):
            shuffled=rng.permutation(joined); diff=shuffled[:len(ad)].mean()-shuffled[len(ad):].mean()
            extreme+=abs(diff)>=abs(observed)-1e-12
        diffs=ad[rng.integers(len(ad),size=(repeats,len(ad)))].mean(axis=1)-cn[rng.integers(len(cn),size=(repeats,len(cn)))].mean(axis=1)
        lo,hi=np.quantile(diffs,[.025,.975]); pooled=np.sqrt(((len(ad)-1)*ad.var(ddof=1)+(len(cn)-1)*cn.var(ddof=1))/(len(ad)+len(cn)-2))
        g=(1-3/(4*(len(ad)+len(cn))-9))*observed/pooled if pooled>0 else float('nan')
        r=dict(metric=metric,difference_ad_minus_cn=float(observed),ci_low=float(lo),ci_high=float(hi),hedges_g=float(g),p_raw=(int(extreme)+1)/(permutations+1))
        for label,x in [('ad',ad),('cn',cn)]:r.update({label+'_'+k:v for k,v in summary(x).items()})
        results.append(r)
    for r,p in zip(results,holm([r['p_raw'] for r in results])):r['p_holm']=float(p)
    return results


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output_dir',default='CADENCE/revision_tracking/analysis/reviewer_statistics')
    ap.add_argument('--bootstrap_repeats',type=int,default=10000); ap.add_argument('--permutations',type=int,default=10000)
    ap.add_argument('--predictions', default='CADENCE/revision_tracking/results/official_test/predictions.csv')
    ap.add_argument('--development_features', default='CADENCE/features/adresso21_text_emo_segments_made_teacher.pkl')
    ap.add_argument('--test_features', default='CADENCE/revision_tracking/features/adresso21_test_text_emo_made_teacher.pkl')
    ap.add_argument('--labels_csv', required=True)
    ap.add_argument('--mmse_csv', required=True)
    args=ap.parse_args(); assert args.bootstrap_repeats>0 and args.permutations>0
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty; preserve previous analysis')
    sources=dict(predictions=args.predictions, development=args.development_features,
        test=args.test_features, labels=args.labels_csv, mmse=args.mmse_csv)
    hashes={k:sha(v) for k,v in sources.items()}
    raw_labels=read_csv(sources['labels']); assert len({r['ID'] for r in raw_labels})==len(raw_labels)
    assert all(r['Dx'] in ['Control','ProbableAD'] for r in raw_labels)
    labels={r['ID']:int(r['Dx']=='ProbableAD') for r in raw_labels}
    comparisons=model_tests(sources['predictions'],labels,np.random.default_rng(42),args.bootstrap_repeats,args.permutations)
    development=load_cache(sources['development'],'train'); test=load_cache(sources['test'],'test')
    assert len(development)==166 and len(test)==71
    assert {r['conv_id'] for r in test}==set(labels)
    assert all(int(r['diagnosis']=='ad')==labels[r['conv_id']] for r in test)
    assert not ({r['conv_id'] for r in development}&set(labels))
    mmse_rows=read_csv(sources['mmse']); assert len({r['adressfname'] for r in mmse_rows})==len(mmse_rows)
    mmse={r['adressfname']:r for r in mmse_rows}
    assert set(mmse)=={r['conv_id'] for r in development}
    for r in development:
        meta=mmse[r['conv_id']];assert meta['dx']==r['diagnosis'];r['mmse']=float(meta['mmse'])
    for r in test:r['mmse']=''
    participants=development+test; stats=group_tests(development,np.random.default_rng(43),args.bootstrap_repeats,args.permutations)
    sample=[]
    for split in ['development','test']:
        for diag in ['cn','ad']:
            selected=[r for r in participants if r['split']==split and r['diagnosis']==diag]
            row=dict(split=split,diagnosis=diag,n_recordings=len(selected),age='not available in analyzed metadata',sex='not available in analyzed metadata')
            row.update({'segments_'+k:v for k,v in summary([r['n_segments'] for r in selected]).items()})
            row.update({'mmse_'+k:v for k,v in summary([r['mmse'] for r in selected if r['mmse']!='']).items()});sample.append(row)
    assert hashes=={k:sha(v) for k,v in sources.items()},'Input changed during analysis'
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'paired_model_tests.csv',comparisons);write_csv(out/'patient_affective_metrics.csv',participants)
    write_csv(out/'affective_group_tests.csv',stats);write_csv(out/'sample_summary.csv',sample)
    write_csv(out/'participant_manifest.csv',[
        {k:r[k] for k in ['conv_id','split','diagnosis','n_segments','n_used_segments','mmse']}
        for r in participants])
    counts=[]
    for split in ['development','test']:
        for diag in ['cn','ad']:
            ns=[r['n_segments'] for r in participants if r['split']==split and r['diagnosis']==diag]
            counts.extend(dict(split=split,diagnosis=diag,n_segments=n,n_recordings=ns.count(n))
                          for n in range(1,max(ns)+1))
    write_csv(out/'segment_histogram_counts.csv',counts)
    protocol=dict(inputs={k:dict(path=v,sha256=hashes[k]) for k,v in sources.items()},seed_model=42,seed_affective=43,
        bootstrap_repeats=args.bootstrap_repeats,permutations=args.permutations,
        bootstrap='participant resampling within diagnosis; paired for model differences; percentile 95% CI',
        pvalue='two-sided absolute statistic; Monte Carlo (extreme+1)/(B+1); exact McNemar and exact prediction swaps when feasible',
        multiplicity='Holm within 3 accuracy tests, 3 weighted-F1 tests, and 5 development affective tests separately',
        analysis_status='revision analyses after test results were observed; not preregistered',
        participant_unit='one conversation ID per analysis row; recording IDs disjoint; original subject mapping requires source evidence',
        speaker_policy='all cached segments, no speaker separation',affective_scope='development only; test affective values descriptive only',
        definitions=dict(neutral='mean neutral probability',entropy='mean segment entropy, natural log',volatility='mean consecutive emo_hidden L2 distance',switch_rate='dominant emotion changes / (T-1)',dwell='mean dominant emotion run length per participant, in segments'),
        group_sd_ddof=1,confounders='unadjusted group association; age, length, ASR and interviewer effects not controlled',
        numpy_version=np.__version__,scipy_version=scipy.__version__,training_performed=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,['development','test']):
        maximum=max(r['n_segments'] for r in participants if r['split']==split)
        bins=np.arange(.5,maximum+1.5,1)
        for diag,color in [('cn','#4477AA'),('ad','#CC6677')]:
            xs=[r['n_segments'] for r in participants if r['split']==split and r['diagnosis']==diag]
            ax.hist(xs,bins=bins,alpha=.6,label=f'{diag.upper()} (n={len(xs)})',color=color)
        ax.set(title=split.capitalize(),xlabel='Segments per recording',ylabel='Recordings');ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'segment_histogram.{ext}',dpi=250)
    plt.close(fig)
    for metric in METRICS:
        fig,ax=plt.subplots(figsize=(4.5,4))
        xs=[[r[metric] for r in development if r['diagnosis']==d and np.isfinite(r[metric])] for d in ['cn','ad']]
        ax.boxplot(xs,tick_labels=['CN','AD'],showmeans=True)
        p=next(r['p_holm'] for r in stats if r['metric']==metric)
        ax.set(xlabel='Diagnosis',ylabel={'dwell':'Mean run length (segments)','entropy':'Mean entropy (nats)','volatility':'Mean consecutive embedding L2 distance','neutral':'Mean neutral probability','switch_rate':'Dominant emotion switching rate'}[metric],title=f'Development; Holm p={p:.3g}')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'{metric}_participant.{ext}',dpi=250)
        plt.close(fig)
    print('Model tests:')
    for r in comparisons:print(r['model_a'],r['model_b'],r['metric'],'difference',round(r['difference_a_minus_b'],4),'Holm p',round(r['p_holm'],6))
    print('Affective tests:')
    for r in stats:print(r['metric'],'AD-CN',round(r['difference_ad_minus_cn'],4),'Holm p',round(r['p_holm'],6))
    print('Saved',out)


if __name__=='__main__':main()
