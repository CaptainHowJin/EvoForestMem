from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
import math
import time

from ..structure_update.contracts import CompletedQuery
from ..structure_update.demand_feedback import (EXTRACT, VERIFY, parse_proposal,
    prompt_sources, segment, verified_witness)
from ..structure_update.demands import DemandWitness
from ..structure_update.feedback import Feedback, source_version
from ..structure_update.requests import response_usage

EXTRACT_CURRENT = EXTRACT + ''' Requirements must be necessary for THIS QUESTION, not desirable additional information. First derive the minimal requirements from the question itself; source contents must not introduce extra requirements. Add question_quote to each requirement: a nonempty exact substring of the question justifying that requirement. Do not add optional implementation details, background facts, or extra verification goals. A missing answer remains unknown; do not replace it with a related easier fact.'''
VERIFY_CURRENT = VERIFY + ''' Also set question_required:true|false for EVERY requirement check. Set true only if the requirement is necessary to answer the supplied question, justified by its question_quote; incidental details are not required. A cited source must actually answer the requirement, not merely repeat the topic. Consider all supplied originals when checking alternative redundancy.'''
EXTRACT_CURRENT = EXTRACT_CURRENT.replace('"need":"...",', '"need":"...","question_quote":"exact substring of the question",')
VERIFY_CURRENT = VERIFY_CURRENT.replace('"key":"r1","alternatives"', '"key":"r1","question_required":true,"alternatives"')


def output_schema(stage):

    def obj(properties):
        return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    def array(items, maximum=None):
        return dict(type='array',items=items,**({'maxItems':maximum} if maximum else {}))
    string, integer, boolean = {'type':'string'}, {'type':'integer','minimum':0}, {'type':'boolean'}
    if stage == 'extract':
        member = obj(dict(source=integer,spans=array(integer)))
        requirement = obj(dict(key=string,need=string,question_quote=string,
            status={'type':'string','enum':['supported','uncertain','unsupported']},
            alternatives=array(obj(dict(members=array(member,40))),2)))
        return obj(dict(requirements={**array(requirement,4),'minItems':1}))
    deletion = obj(dict(member=integer,still_sufficient=boolean))
    alternative = obj(dict(index=integer,sufficient=boolean,time_ok=boolean,speaker_ok=boolean,
        context_ok=boolean,object_ok=boolean,status_ok=boolean,remove_one=array(deletion,40)))
    return obj(dict(checks={**array(obj(dict(key=string,question_required=boolean,
        alternatives=array(alternative,2))),4),'minItems':1}))


class RecallAudit(Feedback):

    def __init__(self, config, client, journal, count_tokens, *, utility_config):
        super().__init__(config, client, journal, count_tokens)
        self.utility_config = utility_config
        self.seconds, self.queries, self.successes, self.audited = 0., set(), 0, 0
        self._pending_witnesses = []

    def _call(self, query, originals, stage, proposal=None):
        body = {k: getattr(query, k) for k in ('question', 'ask_time', 'asker')}
        body['originals'] = prompt_sources(originals)
        if proposal is not None:
            body['requirements'] = proposal
        messages = [dict(role='system', content=EXTRACT_CURRENT if stage == 'extract' else VERIFY_CURRENT),
                    dict(role='user', content=json.dumps(body, ensure_ascii=False, sort_keys=True))]
        tokens = self.client.input_tokens(messages, self.count_tokens)
        limit = 2048
        reference = max(1., self.utility_config.token_reference * sum(self.business.values()))
        spent = sum(self.charges.values())


        price = .01 * math.exp(max(-20., min(20., spent / reference - 1.)))
        expected = (self.successes + 1.) / (self.audited + 2.)
        cost = (tokens + limit) / max(1., reference / max(1, len(self.queries)))
        keys = []
        while True:
            reserved = tokens + limit
            if reserved > self.client.context_window():
                return None, dict(status='context_unverifiable', reserved=reserved)
            payload = dict(model=self.client.model, messages=messages, temperature=0,
                max_tokens=limit, chat_template_kwargs=dict(enable_thinking=False),
                response_format=dict(type='json_schema',json_schema=dict(name='recall_'+stage,
                    strict=True,schema=output_schema(stage))))
            key = self.journal.identity(payload)
            cached = self.journal.cached(payload)
            if (stage == 'extract' and (self.audited or self.charges) and expected < price * cost
                    and cached is None):
                return None, dict(status='soft_budget_deferred', price=price,
                    expected=expected, cost=cost, spent=spent, reference=reference)
            tick = time.perf_counter()
            try:
                response, _, hit = self.journal.complete(payload, role='association_' + stage,
                                                         reserved_tokens=reserved)
                usage = response_usage(response)
                self.charges[key] = sum(usage.values()) if usage else reserved
                keys.append(key)
                self.audit.append(dict(status='response', query=query.identity, stage=stage,
                    request=key, cache_hit=hit, tokens=self.charges[key]))
                choice = response['choices'][0]
                if choice['finish_reason'] == 'length' and limit == 2048:
                    limit = 4096
                    continue
                if choice['finish_reason'] != 'stop':
                    return None, dict(status='incomplete_response', requests=keys)
                return json.loads(choice['message']['content']), dict(status='complete', requests=keys)
            except OSError:
                raise
            except Exception as exc:
                saved = self.journal.cached(payload)
                usage = response_usage((saved or {}).get('response'))
                self.charges[key] = sum(usage.values()) if usage else reserved
                return None, dict(status='failed', requests=keys, error=repr(exc))
            finally:
                self.seconds += time.perf_counter() - tick

    def recognize(self, query, ids, log):
        if not isinstance(query, CompletedQuery):
            raise TypeError('recognition accepts only label-free queries')
        self.queries.add(query.identity)
        if len(ids) > 40 or len(ids) != len(set(ids)):
            raise ValueError('recognition pool must contain at most 40 unique originals')
        originals = segment([dict(id=m, version=source_version(log.get(m)), text=log.get(m).text,
            timestamp=log.get(m).ts, speaker=log.get(m).speaker_key, context=log.get(m).channel_key) for m in ids])
        record = dict(query=query.identity, pool=list(ids), versions={o['id']: o['version'] for o in originals})
        try:
            value, receipt = self._call(query, originals, 'extract')
            record.update(receipt)
            if value is None:
                return None, record
            proposal = parse_proposal(value, originals)
            raw = {r['key']: r for r in value['requirements']}
            for req in proposal:
                quote = raw[req['key']].get('question_quote')
                if not isinstance(quote, str) or not quote.strip() or quote not in query.question:
                    raise ValueError('requirement lacks an exact current-question anchor')
                req['question_quote'] = quote
            if not any(r['alternatives'] for r in proposal):
                record.update(status='unverified', requirements=proposal)
                return None, record
            value, verified = self._call(query, originals, 'verify', proposal)
            keys = [*receipt.get('requests', []), *verified.get('requests', [])]
            record.update(verified, requests=keys)
            if value is None:
                return None, record
            required = {r.get('key') for r in value.get('checks', []) if r.get('question_required') is True}
            for req in proposal:
                if req['key'] not in required:
                    req['status'], req['alternatives'] = 'uncertain', []
            witness, requirements = verified_witness(query, originals, proposal, value, keys, query.ask_time)
            record.update(status='verified' if witness else 'unverified', requirements=requirements)
            if witness:
                self._pending_witnesses.append(asdict(witness))
            return witness, record
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            record.update(status='invalid_feedback', error=repr(exc))
            return None, record
        finally:
            self.audit.append(record)

    def after_batch(self, queries, log, ledger):

        witnesses = [DemandWitness(**w) for w in self._pending_witnesses]
        self._pending_witnesses.clear()
        return witnesses

    def state(self):
        return {**super().state(), 'schema': 'associative-recall-audit-v4.1',
            'utility_config': asdict(self.utility_config), 'seconds': self.seconds,
            'queries': sorted(self.queries), 'successes': self.successes, 'audited': self.audited,
            'pending_witnesses': deepcopy(self._pending_witnesses)}

    def restore(self, value):
        if (value.get('schema') != 'associative-recall-audit-v4.1'
                or value['utility_config'] != asdict(self.utility_config)):
            raise ValueError('fresh associative recall recognition state required')
        super().restore(value)


        for row in self.journal.records():
            if row['role'] in {'association_extract', 'association_verify'}:
                usage = response_usage(row.get('response'))
                self.charges[row['key']] = sum(usage.values()) if usage else row['reserved_tokens']
        self.seconds, self.successes, self.audited = value['seconds'], value['successes'], value['audited']
        self.queries = set(value['queries'])
        self._pending_witnesses = deepcopy(value['pending_witnesses'])


def complete_keys(witness, candidate):
    ids = set(candidate.provenance)
    def quoted(key, mid):
        quotes = [q['text'].strip() for q in witness.quotes if q['requirement'] == key and q['id'] == mid]
        return bool(quotes) and all(q in candidate.context for q in quotes)
    return {r['key'] for r in witness.requirements if r['status'] == 'supported' and
            any(set(g) <= ids and all(quoted(r['key'], m) for m in g) for g in r['groups'])}


def select_current(keep, ranked, witness, mandatory, render, config, allowed):

    original = {s.msg_id: s for s in ranked if s.msg_id in allowed}
    baseline = complete_keys(witness, keep)
    options = [[None, *[g for g in r['groups'] if set(g) <= original.keys()]]
               for r in witness.requirements if r['status'] == 'supported']
    best, examined = keep, 0
    def objective(candidate):
        return (len(complete_keys(witness, candidate)),
                sum(original[m].score for m in candidate.provenance if m in original), -candidate.tokens)
    for choice in product(*options):
        reserved = set(mandatory) | {m for g in choice if g for m in g}
        if len(reserved) > config.max_spans or not reserved <= original.keys():
            continue
        ordered = sorted(original, key=lambda m: (-original[m].score, m))
        ids = [m for m in ordered if m in reserved]
        ids += [m for m in ordered if m not in reserved][:config.max_spans - len(ids)]
        while ids:
            candidate = render(sorted((original[m] for m in ids), key=lambda s: (-s.score, s.msg_id)))
            examined += 1
            if (reserved <= set(candidate.provenance) and candidate.tokens <= config.budget_tokens
                    and candidate.n_spans <= config.max_spans and not candidate.fallback):
                if baseline <= complete_keys(witness, candidate) and objective(candidate) > objective(best):
                    best = candidate
                break
            optional = [m for m in ids if m not in reserved]
            if not optional:
                break
            ids.remove(min(optional, key=lambda m: (original[m].score, m)))
    return best, examined
