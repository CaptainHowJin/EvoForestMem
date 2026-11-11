from pathlib import Path
import hashlib
import json
import os
import tempfile

from .contracts import canonical


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, canonical(value) + b'\n')


def save_object(directory, value):
    data = canonical(value)
    digest = hashlib.sha256(data).hexdigest()
    path = Path(directory) / (digest + '.json')
    if not path.exists():
        atomic_bytes(path, data)
    return {'path': str(path.resolve()), 'sha256': digest}


def load_object(reference):
    if 'offset' in reference or 'length' in reference:
        offset, length = reference.get('offset'), reference.get('length')
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 1:
            raise ValueError('invalid checkpoint object range')
        with Path(reference['path']).open('rb') as stream:
            stream.seek(offset)
            data = stream.read(length)
        if len(data) != length:
            raise ValueError('truncated checkpoint object range')
    else:
        data = Path(reference['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('checkpoint object hash mismatch')
    return json.loads(data)
