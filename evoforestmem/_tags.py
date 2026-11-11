from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from evoforestmem.llm.openai_compat import OpenAICompatProvider
from evoforestmem.log.tags_vocab import LIFECYCLE_LABELS, MARKER_LABELS

def output_schema():
    return {'type': 'object', 'additionalProperties': False, 'required': ['deliverables', 'markers', 'lifecycle', 'lifecycle_confidence'], 'properties': {'deliverables': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 12}, 'markers': {'type': 'array', 'items': {'type': 'string', 'enum': sorted(MARKER_LABELS)}}, 'lifecycle': {'anyOf': [{'type': 'string', 'enum': sorted(LIFECYCLE_LABELS)}, {'type': 'null'}]}, 'lifecycle_confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}}

class JournalProvider(OpenAICompatProvider):

    def __init__(self, spec, journal, experiment):
        super().__init__(spec)
        self.journal = Path(journal)
        self.experiment = experiment

    def client_kwargs(self):
        value = super().client_kwargs()
        value['default_headers']['x-experiment-run'] = self.experiment
        return value

    async def _complete_once(self, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        record = {'msg_id': payload['current']['msg_id'], 'model': self.spec.model, 'stage': 'tag_enrichment', 'experiment': self.experiment, 'messages': messages, 'request_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}
        started = time.perf_counter()
        try:
            result = await super()._complete_once(messages, **kwargs)
            record.update(status='ok', usage=result.usage.to_dict(), finish_reason=result.finish_reason, response_text=result.text)
            return result
        except Exception as error:
            record.update(status='error', error=str(error), usage=None)
            raise
        finally:
            record['elapsed_s'] = time.perf_counter() - started
            with self.journal.open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
