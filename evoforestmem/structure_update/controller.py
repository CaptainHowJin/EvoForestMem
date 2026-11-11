from __future__ import annotations
import random
import time

def valid_summaries(store, trees, delta):
    tree = trees[delta.tree_id]
    for nid in set(delta.modified) | set(delta.ancestors):
        node = tree.nodes[nid]
        if not node.level:
            continue
        card = store.cards.get((tree.tree_id, nid))
        if card is None or card.status not in {'complete', 'inherited'} or card.stale or node.summary_dirty or (tuple(card.structural_children) != tuple(node.child_ids)):
            return False
        for child_id, version in card.child_versions.items():
            child = store.cards.get((tree.tree_id, child_id))
            if child is None or child.version != version:
                return False
    return True

class StructureController:

    def __init__(self, session, config, directory):
        from pathlib import Path
        self.session, self.config, self.directory = (session, config, Path(directory))
        self.history, self.decisions, self.costs = ([], [], [])
        self.rng = random.Random(config.seed)
        self.round = 0

    @property
    def trees(self):
        return self.session.updater.store.trees

    def add_witnesses(self, witnesses):
        seen = {w.identity for w in self.history}
        for witness in witnesses:
            if witness.identity not in seen:
                self.history.append(witness)
                seen.add(witness.identity)
