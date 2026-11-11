from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .contracts import canonical, fingerprint
from .io import atomic_bytes
from .streaming_journal import SCHEMA, StreamJournal


def _encoded(name, data):
    digest = hashlib.sha256(canonical(name) + b'\n' + data).hexdigest()
    header = b'{"name":' + canonical(name) + b',"sha256":' + canonical(digest) + b',"value":'
    return header + data + b'}\n', len(header)


def decode_record(line):
    if not line.endswith(b'\n'):
        raise ValueError('incomplete WAL tail')
    row = json.loads(line)
    if set(row) != {'name', 'sha256', 'value'} or not isinstance(row['name'], str):
        raise ValueError('invalid WAL record')
    data = canonical(row['value'])
    expected, header = _encoded(row['name'], data)
    if line != expected:
        raise ValueError('WAL checksum or canonical encoding mismatch')
    return row['name'], data, header


def read_wal(directory):

    with (Path(directory) / 'records.wal').open('rb') as stream:
        for line in stream:
            name, data, _ = decode_record(line)
            yield name, json.loads(data)


class AppendStreamJournal(StreamJournal):
    def __init__(self, directory, identity):
        self._fd = None
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / 'identity.json').exists():
            raise ValueError('cannot mix file and append stream journals')
        self.path = self.directory / 'records.wal'
        self._index, self._children = {}, defaultdict(set)
        self._failed = False
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            parent = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            self._scan()
            self.put('identity', {'schema': SCHEMA, **identity})
        except BaseException:
            self.close()
            raise

    def _install(self, name, offset, data):
        if name in self._index:
            raise ValueError('duplicate WAL record: ' + name)
        self._index[name] = (offset, len(data), hashlib.sha256(data).hexdigest())
        self._children[name.rpartition('/')[0]].add(name)

    def _scan(self):
        offset = 0
        with self.path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):


                    backup = self.directory / ('torn-' + hashlib.sha256(line).hexdigest() + '.bin')
                    if not backup.exists():
                        atomic_bytes(backup, line)
                    elif backup.read_bytes() != line:
                        raise ValueError('torn-tail backup mismatch')
                    os.ftruncate(self._fd, offset)
                    os.fsync(self._fd)
                    break
                name, data, header = decode_record(line)
                self._install(name, offset + header, data)
                offset += len(line)
        self._end = offset

    def _healthy(self):
        if self._fd is None or self._failed:
            raise RuntimeError('WAL writer closed or failed; reopen for recovery')

    def get(self, name):
        self._healthy()
        ref = self._index.get(name)
        if ref is None:
            return None
        offset, length, digest = ref
        data = os.pread(self._fd, length, offset)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('WAL value checksum mismatch: ' + name)
        return json.loads(data)

    def put(self, name, value):
        return self.put_many([(name, value)])

    def put_many(self, records):

        self._healthy()
        pending, wires, seen, end = [], [], set(), self._end
        for name, value in records:
            if name in seen:
                raise ValueError('duplicate record in WAL group: ' + name)
            seen.add(name)
            data = canonical(value)
            prior = self._index.get(name)
            if prior is not None:
                if hashlib.sha256(data).hexdigest() != prior[2]:
                    raise ValueError('cannot overwrite a durable stream record: ' + name)
                continue
            wire, header = _encoded(name, data)
            pending.append((name, end + header, data))
            wires.append(wire)
            end += len(wire)
        if not pending:
            return
        try:

            if os.fstat(self._fd).st_size != self._end:
                raise ValueError('WAL changed outside its single writer')
            remaining = memoryview(b''.join(wires))
            while remaining:
                count = os.write(self._fd, remaining)
                if count <= 0:
                    raise OSError('short WAL write')
                remaining = remaining[count:]
            os.fsync(self._fd)
            for row in pending:
                self._install(*row)
            self._end = end
        except BaseException:
            self._failed = True
            raise

    def object(self, value):
        digest = hashlib.sha256(canonical(value)).hexdigest()
        name = 'objects/' + digest
        self.put(name, value)
        offset, length, stored = self._index[name]
        return {'path': str(self.path.resolve()), 'offset': offset, 'length': length, 'sha256': stored}

    def decision(self, seq, value, *, commit_prefix=None, payload_sha256=None):

        if (commit_prefix is None) != (payload_sha256 is None):
            raise ValueError('both prefix and payload hash are required for commit')
        data = canonical(value)
        digest = hashlib.sha256(data).hexdigest()
        name = 'objects/' + digest
        prior = self._index.get(name)
        if prior is None:
            _, header = _encoded(name, data)
            offset, length = self._end + header, len(data)
        else:
            offset, length, _ = prior
        ref = {'path': str(self.path.resolve()), 'offset': offset, 'length': length, 'sha256': digest}
        records = [(name, value), (self.step(seq, 'decision'), ref)]
        if commit_prefix is not None:
            records.append((self.step(seq, 'commit'), {'seq': seq, 'decision': ref,
                'prefix_after': fingerprint([commit_prefix, payload_sha256, digest])}))
        self.put_many(records)
        return ref

    def names(self, directory, prefix=''):
        self._healthy()
        start = directory.rstrip('/') + '/' + prefix
        return sorted(name for name in self._children.get(directory, ()) if name.startswith(start))

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        self.close()
