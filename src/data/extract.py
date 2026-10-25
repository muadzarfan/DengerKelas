"""Extract only expected speech files and one consistent DEMAND channel."""
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from .common import ROOT, read_json, write_json, digest, relative, event
from .preprocessing import metadata_rows


def extract_speech(config):
    reports = {}
    for split, source_split in config['fleurs']['split_mapping'].items():
        rows = list(metadata_rows(ROOT/f'data/raw/fleurs/{source_split}.tsv'))
        expected = {row['source_filename'] for row in rows}
        if len(expected) != len(rows):
            raise ValueError(f'Duplicate source filenames in official {split} metadata')
        receipt_path = ROOT/f'data/raw/fleurs/{source_split}_extraction.json'
        old = read_json(receipt_path) if receipt_path.exists() else {}
        if set(old)==expected and all(
            (path:=ROOT/f'data/raw/fleurs/audio/{source_split}/{name}').is_file()
            and path.stat().st_size==record['bytes'] and digest(path)==record['sha256']
            for name,record in old.items()):
            reports[split]={'expected_files':len(rows),'archive_files':len(expected),'newly_extracted':0,'verified_reuse':True}
            event('speech_extracted',split=split,**reports[split])
            continue
        seen, receipts, written = set(), {}, 0
        archive_path = ROOT/f'data/raw/fleurs/{source_split}.tar.gz'
        with tarfile.open(archive_path, mode='r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) != 2 or parts[0] != source_split or parts[1] not in expected:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                name = parts[1]
                if name in seen:
                    raise ValueError('Duplicate archive member')
                seen.add(name)
                output = ROOT/f'data/raw/fleurs/audio/{source_split}/{name}'
                previous = old.get(name,{})
                if output.is_file() and output.stat().st_size == member.size and previous.get('sha256') == digest(output):
                    receipts[name] = previous
                    continue
                output.parent.mkdir(parents=True,exist_ok=True)
                temporary = output.with_suffix('.wav.part')
                with archive.extractfile(member) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src,dst,1024*1024)
                if temporary.stat().st_size != member.size:
                    raise OSError('Incomplete speech extraction')
                temporary.replace(output)
                written += 1
                receipts[name]={'sha256':digest(output),'bytes':output.stat().st_size}
        if seen != expected:
            raise ValueError(f'{split}: missing {len(expected-seen)} or extra source files')
        write_json(receipt_path,receipts)
        reports[split]={'expected_files':len(rows),'archive_files':len(seen),'newly_extracted':written}
        event('speech_extracted',split=split,**reports[split])
    return reports


def extract_noise(config):
    reports=[]
    for split,environment in config['demand']['roles'].items():
        archive_path=ROOT/f'data/raw/demand/{environment}_16k.zip'
        with zipfile.ZipFile(archive_path) as archive:
            members=archive.namelist()
            channels=sorted(n for n in members if PurePosixPath(n).name.startswith('ch') and n.endswith('.wav'))
            selected=[n for n in channels if PurePosixPath(n).name=='ch01.wav']
            if len(selected)!=1 or len(channels)!=16:
                raise ValueError(f'Unexpected channels in {environment}: {channels}')
            source=selected[0]
            output=ROOT/f'data/raw/demand/{environment}/ch01.wav'
            output.parent.mkdir(parents=True,exist_ok=True)
            # Reading through ZipFile validates the selected member's CRC.
            with archive.open(source) as handle:
                data=handle.read()
            import hashlib
            checksum=hashlib.sha256(data).hexdigest()
            if not output.is_file() or digest(output)!=checksum:
                output.write_bytes(data)
            licensing=[]
            for name in members:
                if any(term in PurePosixPath(name).name.lower() for term in ('license','copying','readme')):
                    if archive.getinfo(name).file_size<1024*1024:
                        licensing.append({'member':name,'text':archive.read(name).decode('utf-8',errors='replace')})
            reports.append({'environment':environment,'split':split,'channel':1,'audio_path':relative(output),
                'source_archive':relative(archive_path),'archive_sha256':digest(archive_path),
                'archive_member':source,'available_channels':channels,'audio_sha256':checksum,'archive_license_files':licensing})
            event('noise_extracted',environment=environment,selected_channel=source,available_channels=len(channels))
    return reports


def extract_all():
    config=read_json(ROOT/'configs/data_sources.json')
    speech=extract_speech(config)
    noise=extract_noise(config)
    write_json(ROOT/'results/step3/extraction.json',{'speech':speech,'noise':noise})
    return speech,noise


if __name__=='__main__':
    extract_all()
