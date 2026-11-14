#!/usr/bin/env python
"""Audit saved validation memberships and reproduce the existing split; no training."""
import argparse,csv,json,pickle,hashlib,io,urllib.request
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='CADENCE/revision_tracking/analysis/split_audit')
    parser.add_argument('--features_pkl', default='CADENCE/features/adresso21_text_emo_segments_made_teacher.pkl')
    parser.add_argument('--labels_csv', required=True)
    parser.add_argument('--adresso_root', default='CADENCE/download/ADReSSo21/diagnosis')
    parser.add_argument('--checkpoint_root', default='CADENCE/saved_model')
    parser.add_argument('--source_metadata', help='Optional local official dx-mmse.csv; otherwise download from the source URL')
    args=parser.parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
    cache=Path(args.features_pkl)
    with cache.open('rb') as f: cs=pickle.load(f)['conversations']
    assert len(cs)==166 and all(c['split']=='train' for c in cs)
    ids=[c['conv_id'] for c in cs];diag={c['conv_id']:c['diag_label'] for c in cs}
    assert len(set(ids))==166
    labels=np.array([int(diag[i]=='ad') for i in ids])
    expected={k:set(ids[i] for i in vi) for k,(_,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(ids)),labels),1)}
    configs=[('dual_stream','adresso_seq_cv_par_both_made_teacher','bilstm'),('text_only','adresso_seq_cv_par_both_made_teacher_text','bilstm'),('emotion_only','adresso_seq_cv_par_both_made_teacher_emo_hidden','bilstm'),('bigru','adresso_seq_cv_par_both_made_teacher','bigru'),('transformer','adresso_seq_cv_par_both_made_teacher','transformer'),('pooling','adresso_seq_cv_par_both_made_teacher','pooling')]
    test_path=Path(args.labels_csv);test_rows=read(test_path)
    test={r['ID']:'ad' if r['Dx']=='ProbableAD' else 'cn' for r in test_rows}
    assert len(test)==len(test_rows)==71 and not set(ids)&set(test)
    audio=Path(args.adresso_root)
    paths={p.stem:p for p in (audio/'train/audio').rglob('*.wav')}
    assert set(paths)==set(ids)
    test_paths={p.stem:p for p in (audio/'test-dist/audio').glob('*.wav')}
    assert set(test_paths)==set(test)
    hashes={sha(p):i for i,p in paths.items()}; cross=[i for i,p in test_paths.items() if sha(p) in hashes]
    assert not cross,'Identical audio across development/test'
    manifest=[];counts=[];provenance={str(cache):sha(cache),str(test_path):sha(test_path)}
    for config,directory,model in configs:
        seen=[]
        for k in range(1,6):
            p=Path(args.checkpoint_root)/directory/f'{model}_fold{k}_val_predictions.csv'
            rows=read(p);val=[r['conv_id'] for r in rows]
            assert len(val)==len(set(val)) and set(val)==expected[k],f'Split mismatch: {p}'
            assert all(r['true_label']==diag[r['conv_id']] and int(r['true_id'])==int(r['true_label']=='ad') for r in rows)
            seen+=val;provenance[str(p)]=sha(p)
            for role,selected in [('train',set(ids)-set(val)),('validation',set(val)),('official_test',set(test))]:
                diagnoses=diag if role!='official_test' else test
                for i in sorted(selected):manifest.append(dict(configuration=config,fold=k,recording_id=i,participant_unit=i,diagnosis=diagnoses[i],role=role,participant_unit_basis='benchmark recording proxy; source identity crosswalk unavailable'))
                c=Counter(diagnoses[i] for i in selected)
                counts.append(dict(configuration=config,fold=k,role=role,n=len(selected),ad=c['ad'],cn=c['cn']))
        assert Counter(seen)==Counter(ids),'Validation folds not a unique partition'
    metadata_url='https://talkbank.org/dementia/ADReSSo-2021/demographics/dx-mmse.csv'
    raw=Path(args.source_metadata).read_bytes() if args.source_metadata else urllib.request.urlopen(metadata_url, timeout=60).read()
    metadata=list(csv.DictReader(io.StringIO(raw.decode())))
    official_train={r['adressfname']:r for r in metadata if r['test']=='FALSE'}
    assert set(official_train)==set(ids)
    participant_groups={}
    for r in metadata:
        if r['dataset']=='Pitt':participant_groups.setdefault(r['sid'],[]).append(r)
    issues=[]
    for sid,rs in participant_groups.items():
        if len(rs)<2:continue
        for r in rs:
            folds=[k for k,v in expected.items() if r['adressfname'] in v]
            issues.append(dict(participant_key='Pitt:'+sid,original_recording_id=r['adressfname'],
                source_recording_id=r['id'],split='test' if r['test']=='TRUE' else 'development',
                validation_fold=folds[0] if folds else '',
                cross_official_partition=len({a['test'] for a in rs})>1,
                cross_cv_fold=len({k for k,v in expected.items() if any(a['adressfname'] in v for a in rs)})>1))
    for r in manifest:
        if r['role']!='official_test':
            meta=official_train[r['recording_id']]
            r['participant_unit']=meta['dataset']+':'+(meta['sid'] if meta['dataset']=='Pitt' else meta['id'])
            r['participant_unit_basis']='official Pitt sid' if meta['dataset']=='Pitt' else 'official source recording id; subject identity not independently resolved'
        else:r['participant_unit_basis']='anonymized test recording ID; original-subject crosswalk unavailable'
    out.mkdir(parents=True)
    (out/'official_dx_mmse.csv').write_bytes(raw)
    write(out/'participant_overlap.csv',issues)
    write(out/'split_manifest.csv',manifest);write(out/'fold_distribution.csv',counts)
    report=dict(date='2026-10-12',development_n=166,test_n=71,recording_id_overlap=0,identical_audio_cross_partition=0,
        configs_checked=[c[0] for c in configs],split='StratifiedKFold(5, shuffle=True, random_state=42), original cache order',
        saved_validation_membership_matches_reconstruction=True,validation_coverage='each development record exactly once per configuration',
        train_membership_basis='complement of saved validation IDs, confirmed by training code; historical per-epoch train manifests not saved',
        participant_evidence='Organizers report mitigating repeated speech from same participant; local files lack original participant identity crosswalk. Do not call this an independent identity-level audit.',
        validation_role='same held-out fold used for checkpoint selection and score reporting; no separate inner validation',sources_sha256=provenance,training_performed=False)
    report.update(official_metadata_url=metadata_url, official_metadata_sha256=hashlib.sha256(raw).hexdigest(),
        pitt_recordings=215,pitt_unique_sid=len(participant_groups),
        repeated_pitt_sid=sorted(sid for sid,rs in participant_groups.items() if len(rs)>1),
        cross_official_partition_sid=sorted({r['participant_key'] for r in issues if r['cross_official_partition']}),
        cross_cv_fold_sid=sorted({r['participant_key'] for r in issues if r['cross_cv_fold']}),
        participant_evidence='Official metadata contains repeated Pitt sid across visits; participant independence is contradicted for these records. Exact anonymized test ID crosswalk unavailable.')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in counts:
        if r['configuration']=='dual_stream':print(r)
    print('Recording checks passed; participant overlap detected; see participant_overlap.csv')
if __name__=='__main__':main()
