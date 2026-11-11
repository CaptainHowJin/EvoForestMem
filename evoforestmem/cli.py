from __future__ import annotations
import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def validate_data(root):
    manifest = read(root / 'manifest.json')
    expected = dict(ever_004=560, ever_005=560, ever_010=575, ever_011=595, ever_016=576,
                    group_Finance=175, group_Technology=176, group_Manufacturing=146, group_Healthcare=117)
    if set(manifest['subsets']) != set(expected):
        raise ValueError('Dataset must contain exactly the nine released subsets')
    for subset, count in expected.items():
        spec = manifest['subsets'][subset]; directory = (root / spec['path']).resolve()
        if root.resolve() not in directory.parents:
            raise ValueError('Dataset path escapes root')
        for file, digest in spec['sha256'].items():
            if sha(directory / file) != digest:
                raise ValueError(f'Dataset checksum mismatch: {subset}/{file}')
        qas = jsonl(directory / 'qas.jsonl'); ids = [q['qa_id'] for q in qas]
        if len(ids) != count or len(set(ids)) != count or set(ids) != set(spec['qa_ids']):
            raise ValueError('QA whitelist mismatch: ' + subset)
        messages = jsonl(directory / 'messages.jsonl')
        ever = subset.startswith('ever_')
        lookup = {m['msg_id' if ever else 'msg_node']: m for m in messages}
        for qa in qas:
            for mid in qa.get('evidence_msg_ids' if ever else 'evidence_msg_nodes', []) or []:
                if mid not in lookup:
                    raise ValueError('Unresolved evidence: ' + qa['qa_id'])
                stamp = lookup[mid]['time' if ever else 'timestamp']
                if datetime.fromisoformat(stamp) > datetime.fromisoformat(qa['ask_time']):
                    raise ValueError('Future evidence: ' + qa['qa_id'])
            if str(qa['answer']).strip() in list('ABCD') and set(qa.get('options', {})) != set('ABCD'):
                raise ValueError('Missing MC options: ' + qa['qa_id'])
    return manifest

def scoped_rows(rows, allowed):
    ids = [r['qa_id'] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(allowed):
        raise ValueError('Results contain duplicate or out-of-scope QA')
    return rows

def normalized(messages, qas, subset):
    if subset.startswith('ever_'):
        return messages, qas

    messages = [{**m, 'msg_id': m['msg_node'], 'time': m['timestamp'].replace('T',' '),
                 'speaker': m['author'], 'dialogue': m['content'], 'group': m['project']}
                for m in messages]
    qas = [{**q, 'evidence_msg_ids': q.get('evidence_msg_nodes', []),
            'ask_time': q['ask_time'].replace('T',' ')} for q in qas]
    return messages, qas

async def enrich(log, config, directory):
    from ._tags import JournalProvider, output_schema
    from .llm.provider import ProviderSpec, BudgetPolicy
    from .log.tag_enrich import TagEnrichConfig, enrich_log
    os.environ.setdefault('EVOFORESTMEM_LLM_API_KEY', 'EMPTY')
    client = JournalProvider(ProviderSpec(kind='openai_compat', model=config['model'],
        base_url=config['base_url'], role='tag_enrich', api_key_env='EVOFORESTMEM_LLM_API_KEY',
        max_tokens=384, temperature=0., timeout=180,
        policy=BudgetPolicy(concurrency=config['concurrency'], max_attempts=3),
        extra_body={'chat_template_kwargs': {'enable_thinking':False},
            'response_format': {'type':'json_schema','json_schema':{
                'name':'message_tags','strict':True,'schema':output_schema()}}}),
        directory / 'tag_requests.jsonl', 'evoforestmem-prepare')
    try:
        report = await enrich_log(log, config=TagEnrichConfig(mode='eager', context_window=3,
            request_batch=1, max_inflight=config['concurrency'] * 4),
            cache_path=directory/'tags.jsonl', provider=client)
        if report.fallback > max(1, (len(log)-report.skipped)*.01):
            raise RuntimeError('Tag enrichment failed; refusing silent rule-only preparation')
        return report.to_dict()
    finally:
        await client.aclose()

def prepare(args, config, manifest):
    import numpy as np
    from ._prepare import build_log
    from .index.embedder import LocalEmbedder
    if (args.output_dir/'prepared.json').exists():
        raise ValueError('Prepared output exists; choose a new output directory')
    dest = args.output_dir / 'runtime'; spec = manifest['subsets'][args.subset]
    source = args.data_root / spec['path']
    messages, qas = normalized(jsonl(source/'messages.jsonl'), jsonl(source/'qas.jsonl'), args.subset)
    log, channels, speakers, id_map = build_log(messages)
    preparation_identity = {'subset':args.subset,'config':config,'dataset':sha(args.data_root/'manifest.json')}
    identity_path = args.output_dir/'preparation_identity.json'
    if identity_path.exists() and read(identity_path) != preparation_identity:
        raise ValueError('Preparation cache identity mismatch')
    write(identity_path, preparation_identity)
    directory = dest/'data/log'/args.subset; directory.mkdir(parents=True,exist_ok=True)
    tag_report = asyncio.run(enrich(log, config, directory))
    log.save(directory/'messages.jsonl');channels.save(directory/'channels.json');speakers.save(directory/'speakers.json')
    write(directory/'manifest.json',dict(source=spec['path'],messages=len(log),qa=len(qas),tag_enrich=tag_report))
    index = dest/'data/index'/args.subset; index.mkdir(parents=True,exist_ok=True)
    embedder = LocalEmbedder(config['embedding_model'],device=config['device'], cache_dir=index/'embeddings_cache')
    ordered = sorted(log,key=lambda m:(m.ts,m.seq,m.msg_id))
    vectors = [embedder.encode([m.text for m in ordered[i:i+64]],kind='document') for i in range(0,len(ordered),64)]
    emb = index/'embeddings';emb.mkdir(exist_ok=True)
    np.save(emb/'vectors.npy',np.concatenate(vectors))
    write(emb/'ids.json',[m.msg_id for m in ordered]);write(index/'dataset_id_map.json',id_map)
    (index/'qa_with_time.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in qas))
    write(args.output_dir/'prepared.json',{'subset':args.subset,'dataset_sha256':sha(args.data_root/'manifest.json'),'config':config,
        'files':{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and 'embeddings_cache' not in p.parts and p.name!='tag_requests.jsonl'}})

def run(args, config, manifest):
    from . import _stream_driver as driver
    from .structure_update.runtime import ExperimentHooks
    from .structure_update.contracts import StructureUpdateConfig, fingerprint
    from .structure_update.streaming_policy import StreamingConfig
    from .structure_update.online_utility import OnlineUtilityConfig
    from .associations.contracts import AssociationConfig
    prepared = read(args.output_dir/'prepared.json')
    if prepared['subset'] != args.subset or prepared['dataset_sha256'] != sha(args.data_root/'manifest.json') or prepared['config'] != config:
        raise ValueError('Preparation configuration or data differs')
    runtime = args.output_dir/'runtime'
    for name,digest in prepared['files'].items():
        if sha(runtime/name) != digest:raise ValueError('Prepared input changed: '+name)
    allowed=manifest['subsets'][args.subset]['qa_ids']
    qas=scoped_rows(jsonl(runtime/'data/index'/args.subset/'qa_with_time.jsonl'),allowed)
    if args.limit_qa:qas=qas[:args.limit_qa]

    for name in ['results.json','retrieval.partial.json']:
        p=args.output_dir/name
        if p.exists():scoped_rows(read(p)['rows'],[q['qa_id'] for q in qas])
    identity={'subset':args.subset,'dataset':prepared['dataset_sha256'],'config':config,'limit_qa':args.limit_qa,
              'code':{str(p.relative_to(PACKAGE)):sha(p) for p in PACKAGE.rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.json'}}}
    record=args.output_dir/'run.json'
    if record.exists():
        if not args.resume or read(record)!=identity:raise ValueError('Run exists; resume requires identical inputs/configuration/code')
    else:write(record,identity)
    hooks=ExperimentHooks(args.output_dir,StructureUpdateConfig(structure_update_mode='stream_verified_v1',seed=1729,replay_workers=2),
        answer_base_url=config['base_url'],answer_model=config['model'],judge_base_url=config['judge_base_url'],judge_model=config['judge_model'],
        qa_rows=qas,run_identity=fingerprint(identity),
        association_config=AssociationConfig(mode='verified_v1',navigation='recall_v4',learning_policy='associative_recall_v4',seed=1729),
        streaming_config=StreamingConfig(proposal_policy='utility_v1',citation_repair='bounded_v1',journal_backend='append_v2'),
        utility_config=OnlineUtilityConfig(mode='online',token_reference=.05))
    driver.REPO=runtime
    driver.EMBEDDING_MODEL=config["embedding_model"]
    summary, rows = driver.run(args.subset, limit_qa=args.limit_qa,
        overrides={'s4': {'max_spans': 10, 'budget_tokens': 1500}, 'fallback': {'k': 10},
                   'navigation': {'strategy': 'hierarchical_v1'}},
        artifact_routing='tag', artifact_min_messages=2,
        embedding_device=config['device'], qwen_tokenizer=config['tokenizer'],
        qwen_base_url=config['base_url'], qwen_model=config['model'],
        routing_summary_config_path=PACKAGE/'config/incremental_compact_v1.yaml',
        summary_concurrency=config['concurrency'], summary_refresh_policy='change_gated',
        summary_profile='compact_v1', summary_refresh_interval=4096,
        progress_out=args.output_dir/'summary.progress.json',
        summary_cache_dir=args.output_dir/'summary_cache',
        routing_summary_out=args.output_dir/'summary.routing_summaries.json',
        resume_from_progress=args.resume, point_observer=hooks)
    write(args.output_dir/'summary.json', summary)
    write(args.output_dir/'summary.per_question.json', {'rows': rows})
    result = hooks.finish()
    if result is not None:
        result['subset'] = args.subset
        for row in result['rows']:
            row['subset'] = args.subset
        write(args.output_dir/'results.json', result)

def judge(args,config,manifest):
    from .structure_update.requests import RequestJournal
    from .structure_update.runtime import judge_results
    run_record=read(args.output_dir/'run.json')
    if run_record['subset']!=args.subset or run_record['dataset']!=sha(args.data_root/'manifest.json'):
        raise ValueError('Result dataset identity mismatch')
    path=args.output_dir/'results.json'
    results = read(path)
    if results.get('subset') != args.subset:
        raise ValueError('Result subset mismatch')
    scoped_rows(results['rows'],manifest['subsets'][args.subset]['qa_ids'])
    if any(row.get('subset') != args.subset for row in results['rows']):
        raise ValueError('Result row subset mismatch')
    journal=RequestJournal(args.output_dir/'judging_requests',config['judge_base_url'])
    try:judge_results(path,journal,config['judge_model'])
    finally:journal.close()

class Tee:
    def __init__(self, console, log):self.console,self.log=console,log
    def write(self,text):self.console.write(text);self.log.write(text);self.log.flush();return len(text)
    def flush(self):self.console.flush();self.log.flush()

def main(argv=None):
    parser=argparse.ArgumentParser(description='Standalone data preparation, streaming execution and scoped evaluation.')
    parser.add_argument('command',choices=['validate-data','prepare','run','judge'])
    parser.add_argument('--data-root',type=Path,default=Path('datasets'))
    parser.add_argument('--subset')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--config',type=Path,default=PACKAGE/'config/runtime.json')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit-qa',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    manifest=validate_data(args.data_root)
    if args.command=='validate-data':
        print('Validated 3,480 QA across 9 subsets.');return 0
    if args.subset not in manifest['subsets'] or not args.output_dir:parser.error('--subset and --output-dir are required')
    if args.limit_qa is not None and args.limit_qa<=0:parser.error('--limit-qa must be positive')
    config=read(args.config)
    if args.dry_run:print(json.dumps({'command':args.command,'subset':args.subset,'config':config},indent=2));return 0
    args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    logdir=args.output_dir/'execution';logdir.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');record={'command':sys.argv,'started_at':stamp,'config':config,
        'code_sha256':{str(p.relative_to(PACKAGE)):sha(p) for p in PACKAGE.rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.json'}},'exit_status':None}
    try:
        with (logdir/(stamp+'.log')).open('w') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:globals()[args.command](args,config,manifest)
            except BaseException:
                import traceback
                traceback.print_exc();raise
        record['exit_status']=0
    except BaseException:
        record['exit_status']=1;raise
    finally:
        record['ended_at']=datetime.now(timezone.utc).isoformat();write(logdir/(stamp+'.json'),record)
    return 0
