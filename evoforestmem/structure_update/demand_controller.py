from copy import deepcopy
from .controller import StructureController
from .feedback import source_version, applicable

class DemandController(StructureController):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluated, self.last_probes, self.proposal_audit = ({}, {}, {})

    def legal_history(self, known_at):
        log, ledger = (self.session.log, self.session.updater.ledger)
        legal = []
        for original in self.history:
            if original.ask_time > known_at or original.verified_at > known_at:
                continue
            witness = deepcopy(original)
            for r in witness.requirements:
                r['groups'] = [g for g in r['groups'] if all((mid in log and source_version(log.get(mid)) == witness.sources.get(mid) and applicable(log, ledger, mid, witness.ask_time) and applicable(log, ledger, mid, known_at) for mid in g))]
                if r['status'] == 'supported' and (not r['groups']):
                    r['status'] = 'uncertain'
            if witness.groups:
                legal.append(witness)
        return sorted(legal, key=lambda w: (w.ask_time, w.identity))
