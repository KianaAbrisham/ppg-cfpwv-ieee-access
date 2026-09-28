"""Download, verify and convert official PWDB v0.2 CSVs by explicit subject ID.

PWDB contains simulated physiology, not recordings from human participants.
This converter preserves amplitudes, sample order and trailing empty cells.
It applies no filtering, resampling, normalization or cropping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import numpy as np
import pandas as pd

RECORD='3275625'
FILES={'PWs_csv.zip':'81067f96d6078bbbb5cd9fce5d73f5bd',
       'pwdb_haemod_params.csv':'43e1244665e6cee6b77501102404b70a',
       'pwdb_pw_indices.csv':'87946898d39d5a6d894901ad39f6b546'}
SITES={'digital':'Digital','radial':'Radial','brachial':'Brachial'}


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def obtain(source,download):
    source.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name,expected in FILES.items():
        path=source/name
        url=f'https://zenodo.org/api/records/{RECORD}/files/{name}/content'
        if not path.exists():
            if not download:raise FileNotFoundError(f'{path}; use --download to obtain the official file.')
            partial=path.with_suffix(path.suffix+'.part')
            print(f'Downloading {name}',flush=True)
            req=urllib.request.Request(url,headers={'User-Agent':'PWDB-reproducibility-research/1.0'})
            with urllib.request.urlopen(req,timeout=120) as response,partial.open('wb') as out:
                shutil.copyfileobj(response,out)
            if digest(partial,'md5')!=expected:raise ValueError(f'Official MD5 mismatch: {name}')
            partial.replace(path)
        if digest(path,'md5')!=expected:raise ValueError(f'Official MD5 mismatch: {name}')
        manifest.append({'name':name,'url':url,'md5':expected,'sha256':digest(path),'bytes':path.stat().st_size})
    return manifest


def table(path):
    frame=pd.read_csv(path)
    frame.columns=frame.columns.str.strip()
    if frame.columns.duplicated().any() or 'Subject Number' not in frame:raise ValueError(f'Invalid schema: {path}')
    ids=frame['Subject Number'].to_numpy(float)
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all() or (ids<=0).any():
        raise ValueError(f'Invalid subject IDs: {path}')
    frame['Subject Number']=ids.astype(np.int64)
    if frame['Subject Number'].duplicated().any():raise ValueError(f'Duplicate IDs: {path}')
    if set(frame['Subject Number'])!=set(range(1,4375)):raise ValueError(f'Unexpected subject set: {path}')
    return frame.set_index('Subject Number').sort_index()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output directory; existing results are protected.')
    inputs=obtain(a.source,a.download)
    haem=table(a.source/'pwdb_haemod_params.csv');idx=table(a.source/'pwdb_pw_indices.csv')
    if not np.array_equal(haem['age [years]'],idx['Age']):raise ValueError('Age mismatch after ID alignment.')
    targets=haem['PWV_cf [m/s]'].to_numpy(float);ages=haem['age [years]'].to_numpy(int)
    if not np.isfinite(targets).all() or (targets<=0).any():raise ValueError('Invalid cf-PWV target.')
    av,ac=np.unique(ages,return_counts=True)
    if not np.array_equal(av,[25,35,45,55,65,75]) or not np.array_equal(ac,np.full(6,729)):
        raise ValueError('Unexpected age distribution.')
    ieee=a.output/'ieee'/'data';ieee.mkdir(parents=True)
    dl=a.output/'deep-learning';dl.mkdir()
    ids=haem.index.to_numpy()
    pd.DataFrame({'subject_id':ids,'cfpwv_m_s':targets}).to_csv(dl/'targets_cf.csv',index=False)
    pd.DataFrame({'subject_id':ids,'age':ages}).to_csv(dl/'targets_age.csv',index=False)
    haem[['PWV_cf [m/s]']].reset_index().to_csv(ieee/'PWV.csv',index=False)
    sites={}
    with zipfile.ZipFile(a.source/'PWs_csv.zip') as archive:
        for site,label in SITES.items():
            member=f'csv/PWs_{label}_PPG.csv';raw_path=a.source/f'PWs_{label}_PPG.csv'
            raw_path.write_bytes(archive.read(member));raw=table(raw_path)
            if raw.columns.tolist()!=[f'pt{i}' for i in range(1,len(raw.columns)+1)]:
                raise ValueError(f'{site}: unordered samples.')
            x=raw.to_numpy(float);lengths=[]
            for row in x:
                present=np.flatnonzero(~np.isnan(row))
                if not len(present):raise ValueError(f'{site}: empty waveform.')
                n=int(present[-1]+1)
                if not np.isfinite(row[:n]).all() or np.ptp(row[:n])<=1e-8:
                    raise ValueError(f'{site}: internal gap, infinity or constant waveform.')
                lengths.append(n)
            dest=dl/site;dest.mkdir()
            signals=pd.DataFrame(x,columns=[f's{i:04d}' for i in range(x.shape[1])]);signals.insert(0,'subject_id',ids)
            signals.to_csv(dest/'signals.csv',index=False)
            check=pd.read_csv(dest/'signals.csv')
            np.testing.assert_array_equal(check.subject_id.to_numpy(),ids)
            np.testing.assert_allclose(check.iloc[:,1:].to_numpy(),x,rtol=1e-14,atol=1e-15,equal_nan=True)
            raw.reset_index().to_csv(ieee/f'PWs_{label}_PPG.csv',index=False)
            selected=idx[['Age']+[c for c in idx if c.startswith(label+'_')]]
            idxname={'digital':'digfeatures.csv','radial':'radfeature.csv','brachial':'brachfeatures.csv'}[site]
            selected.reset_index().to_csv(ieee/idxname,index=False)
            t=selected[f'{label}_PPGsys_T'].to_numpy(float);finite=np.isfinite(t)
            if not ((t[finite]>=0)&(t[finite]<=(np.array(lengths)[finite]-1)/500)).all():
                raise ValueError(f'{site}: fiducial time outside waveform.')
            pd.DataFrame({'subject_id':ids,'n_samples':lengths}).to_csv(dest/'signal_lengths.csv',index=False)
            sites[site]={'subjects':len(ids),'min_samples':min(lengths),'max_samples':max(lengths),
                'archive_member':member,'raw_sha256':digest(raw_path),'signals_sha256':digest(dest/'signals.csv'),
                'nonfinite_systolic_times':int((~finite).sum())}
    result={'source_record':RECORD,'version':'0.2.0','source_kind':'in_silico_virtual_healthy_adults',
        'sampling_rate_hz':500,'signal_units':'arbitrary units','cfpwv_units':'m/s',
        'subject_alignment':'explicit Subject Number, sorted by ID; no positional joins',
        'transformations':'strip header whitespace; rename columns; preserve all waveform values and trailing NaNs',
        'inputs':inputs,'sites':sites,'age_counts':{str(v):int(c) for v,c in zip(av,ac)},
        'cfpwv_range_m_s':[float(targets.min()),float(targets.max())],'recommended_length':487,
        'paper_reproduction_verified':False}
    result['outputs']={str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*.csv')}
    (a.output/'conversion_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'sites':sites,'age_counts':result['age_counts']},indent=2),flush=True)


if __name__=='__main__':main()
