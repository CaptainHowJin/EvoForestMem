from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

from .contracts import fingerprint
from .io import write_json


def response_usage(response):

    usage = response.get('usage') if isinstance(response, dict) else None
    names = ('prompt_tokens', 'completion_tokens')
    if not isinstance(usage, dict) or not all(type(usage.get(k)) is int and usage[k] >= 0 for k in names):
        return {}
    return {k: usage[k] for k in names}


class RequestJournal:
    def __init__(self, directory, base_url, *, sources=(), cache_only=False, timeout=600,
                 append=False):
        self.directory, self.base_url = Path(directory), base_url.rstrip('/')
        self.sources = tuple(Path(p) for p in sources)
        self.cache_only, self.timeout = cache_only, timeout
        self._lock, self._locks = threading.Lock(), {}
        self.events = []
        self.append = append
        self._storage_lock = threading.RLock()
        self._storage_ready = threading.Condition(self._storage_lock)
        self._writes, self._flushing, self._storage_error = [], False, None
        self._writer = None
        self._append_records = {}

    def _appended(self, directory):

        with self._storage_lock:
            if directory not in self._append_records:
                from .streaming_wal import read_wal
                rows = {}
                if (directory / 'append' / 'records.wal').exists():
                    for name, row in read_wal(directory / 'append'):
                        if name.startswith('requests/'):
                            if name != 'requests/' + row['key'] + '/' + row['status']:
                                raise ValueError('request journal key/status mismatch')
                            rows[row['key']] = row
                self._append_records[directory] = rows
            return self._append_records[directory]

    def _store(self, row):
        if not self.append:
            write_json(self.directory / (row['key'] + '.json'), row)
            return
        entry = {'row': dict(row), 'done': False}
        with self._storage_ready:
            if self._storage_error is not None:
                raise RuntimeError('request WAL failed; reopen for recovery') from self._storage_error
            self._writes.append(entry)
        while True:
            with self._storage_ready:
                if self._storage_error is not None:
                    raise self._storage_error
                if entry['done']:
                    return
                if self._flushing:
                    self._storage_ready.wait()
                    continue
                batch, self._writes = self._writes, []
                self._flushing = True
            error = None
            try:
                from .streaming_wal import AppendStreamJournal
                if self._writer is None:
                    self._writer = AppendStreamJournal(self.directory / 'append',
                                                        {'purpose': 'model-requests-v1'})

                    import os
                    for path in (self.directory, self.directory.parent):
                        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                self._writer.put_many([('requests/' + item['row']['key'] + '/' +
                    item['row']['status'], item['row']) for item in batch])
            except BaseException as exc:
                error = exc
            with self._storage_ready:
                if error is None:
                    for item in batch:

                        self._appended(self.directory)[item['row']['key']] = item['row']
                        item['done'] = True
                else:
                    self._storage_error = error
                self._flushing = False
                self._storage_ready.notify_all()

    def close(self):
        with self._storage_ready:
            while self._flushing:
                self._storage_ready.wait()
            if self._writer is not None:
                self._writer.close()
                self._writer = None

    def identity(self, payload, replicate=0):

        return fingerprint({'payload': payload, 'replicate': replicate})

    def cached(self, payload, replicate=0):
        key = self.identity(payload, replicate)
        for directory in (self.directory, *self.sources):
            path = directory / (key + '.json')
            row = self._appended(directory).get(key)
            if row is None and path.exists():
                row = json.loads(path.read_text())
            if row is not None:
                if row.get('payload') != payload or row.get('replicate') != replicate:
                    raise ValueError('full request cache mismatch')
                return row
        return None

    def complete(self, payload, *, role, replicate=0, reserved_tokens=0):
        key = self.identity(payload, replicate)
        with self._lock:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            cached = self.cached(payload, replicate)
            if cached is not None:
                if cached['status'] != 'complete':
                    raise RuntimeError('recorded request failure: ' + cached.get('error', cached['status']))
                return cached['response'], key, True
            if self.cache_only:
                raise LookupError('full request cache miss: ' + key)
            row = {'key': key, 'payload': payload, 'replicate': replicate, 'role': role,
                   'status': 'started', 'reserved_tokens': reserved_tokens}
            self._store(row)
            started = time.perf_counter()
            try:
                req = Request(self.base_url + '/chat/completions',
                              json.dumps(payload, ensure_ascii=False).encode(),
                              {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ.get('EVOFORESTMEM_JUDGE_API_KEY' if role == 'judge' else 'EVOFORESTMEM_LLM_API_KEY', 'EMPTY')})
                with urlopen(req, timeout=self.timeout) as response:
                    value = json.load(response)
                row['response'] = value
                if not isinstance(value, dict) or value.get('model') != payload['model']:
                    raise ValueError('response model differs from the frozen request')
                if not response_usage(value):
                    raise ValueError('missing actual token usage')
                if value['choices'][0]['finish_reason'] not in {'stop', 'length'}:
                    raise ValueError('incomplete response: ' + str(value['choices'][0]['finish_reason']))
                row['status'] = 'complete'
            except BaseException as exc:
                row.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
                raise
            finally:
                row['elapsed_seconds'] = time.perf_counter() - started
                self._store(row)
                self.events.append({'key': key, 'cache_hit': False, 'role': role,
                    'status': row['status'], 'usage': response_usage(row.get('response')),
                    'elapsed_seconds': row['elapsed_seconds']})
            return value, key, False

    def records(self):
        rows = {p.stem: json.loads(p.read_text()) for p in self.directory.glob('*.json')}
        rows.update(self._appended(self.directory))
        return [rows[key] for key in sorted(rows)]


class SummaryJournalClient:

    def __init__(self, client, journal):
        self.client, self.journal = client, journal
        self.role = 'business_summary'

    @property
    def events(self):
        return self.journal.events

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, messages, *, max_tokens, citation_count=None):
        payload = {'model': self.model, 'messages': messages, 'temperature': 0,
                   'max_tokens': max_tokens, 'chat_template_kwargs': {'enable_thinking': False},
                   'response_format': {'type': 'json_object'}}
        if citation_count is not None:
            if type(citation_count) is not int or citation_count < 1:
                raise ValueError('positive citation source count required')
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'bounded_summary_citations_v1', 'strict': True,
                'schema': {'type': 'object', 'properties': {
                    'text': {'type': 'string', 'minLength': 1},
                    'citations': {'type': 'array', 'minItems': 1,
                                  'items': {'type': 'integer', 'enum': list(range(citation_count))}}},
                    'required': ['text', 'citations'], 'additionalProperties': False}}}
        response, key, hit = self.journal.complete(payload,
            role='summary_citation_repair' if citation_count is not None else self.role)
        choice = response['choices'][0]
        usage = dict(response['usage'])
        usage['cached_tokens'] = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        return {'text': choice['message']['content'], 'usage': usage, 'model': response['model'],
                'finish_reason': choice['finish_reason']}
