from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from .s0_analyse import QuerySpec
from .s1_recall import S1Config, S1Plan
from .s2_scan import S2Config, S2Result
from .s3_join import DEFAULT_CONFIG as DEFAULT_JOIN_CONFIG
from .s3_join import JoinConfig, JoinResult
from .s4_assemble import Assembled, SectionMode
REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO / 'config' / 'evoforestmem.yaml'
FallbackReason = Literal['no_trees', 'low_tree', 'few_spans', 'budget']
DepthPolicy = Literal['fixed', 'spec']
S2Source = Literal['auto', 'dense', 'leaf', 'off']
S2LeafWindow = Literal['prefix', 'centered']

@dataclass(frozen=True, slots=True)
class PipelineConfig:

    retrieval_strategy: str = 'legacy'
    navigation_nodes: int = 128
    navigation_sources: int = 512
    navigation_view_floor: int = 16
    navigation_direct: int = 32
    navigation_expand_once: bool = True
    navigation_fallback: bool = True
    s1: S1Config = field(default_factory=S1Config)
    s2: S2Config = field(default_factory=S2Config)
    s3: JoinConfig = DEFAULT_JOIN_CONFIG
    use_s2: bool = True
    use_s3: bool = True
    s2_source: S2Source = 'dense'
    s2_max_leaves: int = 0
    s2_leaf_window: S2LeafWindow = 'prefix'
    s2_leaf_agg: Literal['max', 'mean', 'sum'] = 'max'
    budget_tokens: int = 3000
    max_spans: int | None = 20
    s1_span_budget: int = 14
    expansion_span_budget: int = 6
    max_depth: int = 2
    depth_policy: DepthPolicy = 'spec'
    style: Literal['plain', 'structured'] = 'structured'
    header_budget_frac: float = 0.35
    section_headers: SectionMode = 'on_change'
    lifecycle_marks: bool = True
    marker_marks: bool = True
    duration_headers: bool = True
    leaf_up_headers: bool = True
    validity_headers: bool = True
    validity_penalty: float = 0.25
    sufficiency_header: bool = True
    rank_numbers: bool = True
    render_display_names: bool = True
    render_seconds: bool = True
    persona_cards: bool = False
    actor_evidence_spans: int = 0
    norm_spans: int = 3
    validity_overlay: bool = True
    bridge_hop: bool = False
    bridge_span_budget: int = 2
    min_spans: int = 3
    min_tree_score: float = 0.0
    fallback_k: int = 20
    fallback_enabled: bool = True
    summaries: Literal['off', 'lazy', 'eager'] = 'eager'
    corpus_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {'navigation': {'strategy': self.retrieval_strategy, 'nodes': self.navigation_nodes, 'sources': self.navigation_sources, 'view_floor': self.navigation_view_floor, 'direct': self.navigation_direct, 'expand_once': self.navigation_expand_once, 'fallback': self.navigation_fallback}, 's1': self.s1.to_dict(), 's2': {'mode': self.s2.mode, 'm': self.s2.m, 'refine_to': self.s2.refine_to, 'min_level': self.s2.min_level, 'epsilon': self.s2.epsilon, 'extra_leaves_per_path': self.s2_max_leaves, 'extra_leaf_window': self.s2_leaf_window}, 's3': {'k_anchor': self.s3.k_anchor, 'k_pivot': self.s3.k_pivot, 'max_pairs': self.s3.max_pairs, 'pivot_lifecycle': self.s3.pivot_lifecycle}, 'stages': {'s2': self.use_s2, 's3': self.use_s3, 's2_source': self.s2_source}, 'expansion': {'s1_span_budget': self.s1_span_budget, 'span_budget': self.expansion_span_budget, 'bridge_hop': self.bridge_hop, 'bridge_span_budget': self.bridge_span_budget}, 's4': {'budget_tokens': self.budget_tokens, 'max_spans': self.max_spans, 'max_depth': self.max_depth, 'depth_policy': self.depth_policy, 'style': self.style, 'header_budget_frac': self.header_budget_frac, 'section_headers': self.section_headers, 'lifecycle_marks': self.lifecycle_marks, 'marker_marks': self.marker_marks, 'duration_headers': self.duration_headers, 'leaf_up_headers': self.leaf_up_headers, 'sufficiency_header': self.sufficiency_header, 'rank_numbers': self.rank_numbers, 'render_display_names': self.render_display_names, 'render_seconds': self.render_seconds, 'persona_cards': self.persona_cards, 'actor_evidence_spans': self.actor_evidence_spans, 'norm_spans': self.norm_spans, 'validity_overlay': self.validity_overlay}, 'fallback': {'min_spans': self.min_spans, 'min_tree_score': self.min_tree_score, 'k': self.fallback_k, 'enabled': self.fallback_enabled}, 'summaries': self.summaries, 'corpus_profile': self.corpus_profile}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> 'PipelineConfig':

        d = dict(d or {})
        s2_block = dict(d.get('s2') or {})
        s3_block = dict(d.get('s3') or {})
        stages = dict(d.get('stages') or {})
        s4 = dict(d.get('s4') or {})
        expansion = dict(d.get('expansion') or {})
        fb = dict(d.get('fallback') or {})
        navigation = dict(d.get('navigation') or {})
        _check(navigation, {'strategy', 'nodes', 'sources', 'view_floor', 'direct', 'expand_once', 'fallback'}, 'navigation')
        validity_block_cfg = dict(d.get('validity') or {})
        _check(s2_block, {'mode', 'm', 'beam_width', 'refine_to', 'max_leaves_per_path', 'missing_score', 'min_level', 'epsilon', 'enabled', 'source', 'leaf_agg', 'extra_leaves_per_path', 'extra_leaf_window'}, 's2')
        _check(stages, {'s2', 's3', 's2_source'}, 'stages')
        _check(s4, {'budget_tokens', 'max_spans', 'max_depth', 'depth_policy', 'style', 'header_budget_frac', 'section_headers', 'lifecycle_marks', 'marker_marks', 'duration_headers', 'leaf_up_headers', 'sufficiency_header', 'rank_numbers', 'render_display_names', 'render_seconds', 'persona_cards', 'actor_evidence_spans', 'norm_spans', 'validity_overlay'}, 's4')
        _check(expansion, {'s1_span_budget', 'span_budget', 'bridge_hop', 'bridge_span_budget'}, 'expansion')
        _check(fb, {'enabled', 'min_spans', 'min_tree_score', 'k'}, 'fallback')
        base = cls()
        kw: dict[str, Any] = {'retrieval_strategy': str(navigation.get('strategy', base.retrieval_strategy)), **{f'navigation_{key}': navigation.get(key, getattr(base, f'navigation_{key}')) for key in ('nodes', 'sources', 'view_floor', 'direct', 'expand_once', 'fallback')}, 's1': _apply(S1Config(), d.get('s1'), 's1'), 's2': S2Config.from_dict(s2_block), 's3': _apply(DEFAULT_JOIN_CONFIG, s3_block, 's3', skip={'enabled'}), 'use_s2': bool(stages.get('s2', s2_block.get('enabled', base.use_s2))), 'use_s3': bool(stages.get('s3', s3_block.get('enabled', base.use_s3))), 's2_source': str(stages.get('s2_source', s2_block.get('source', base.s2_source))), 's2_max_leaves': int(s2_block.get('extra_leaves_per_path', base.s2_max_leaves)), 's2_leaf_window': str(s2_block.get('extra_leaf_window', base.s2_leaf_window)), 's2_leaf_agg': str(s2_block.get('leaf_agg', base.s2_leaf_agg)), 'budget_tokens': int(s4.get('budget_tokens', base.budget_tokens)), 'max_spans': None if s4.get('max_spans', base.max_spans) is None else int(s4['max_spans'] if 'max_spans' in s4 else base.max_spans), 's1_span_budget': int(expansion.get('s1_span_budget', base.s1_span_budget)), 'expansion_span_budget': int(expansion.get('span_budget', base.expansion_span_budget)), 'max_depth': int(s4.get('max_depth', base.max_depth)), 'depth_policy': str(s4.get('depth_policy', base.depth_policy)), 'style': str(s4.get('style', base.style)), 'header_budget_frac': float(s4.get('header_budget_frac', base.header_budget_frac)), 'section_headers': str(s4.get('section_headers', base.section_headers)), 'lifecycle_marks': bool(s4.get('lifecycle_marks', base.lifecycle_marks)), 'marker_marks': bool(s4.get('marker_marks', base.marker_marks)), 'duration_headers': bool(s4.get('duration_headers', base.duration_headers)), 'validity_headers': bool(validity_block_cfg.get('headers', base.validity_headers)), 'validity_penalty': float(validity_block_cfg.get('penalty', base.validity_penalty)), 'sufficiency_header': bool(s4.get('sufficiency_header', base.sufficiency_header)), 'rank_numbers': bool(s4.get('rank_numbers', base.rank_numbers)), 'render_display_names': bool(s4.get('render_display_names', base.render_display_names)), 'render_seconds': bool(s4.get('render_seconds', base.render_seconds)), 'persona_cards': bool(s4.get('persona_cards', base.persona_cards)), 'actor_evidence_spans': int(s4.get('actor_evidence_spans', base.actor_evidence_spans)), 'norm_spans': int(s4.get('norm_spans', base.norm_spans)), 'validity_overlay': bool(s4.get('validity_overlay', base.validity_overlay)), 'leaf_up_headers': bool(s4.get('leaf_up_headers', base.leaf_up_headers)), 'bridge_hop': bool(expansion.get('bridge_hop', base.bridge_hop)), 'bridge_span_budget': int(expansion.get('bridge_span_budget', base.bridge_span_budget)), 'min_spans': int(fb.get('min_spans', base.min_spans)), 'min_tree_score': float(fb.get('min_tree_score', base.min_tree_score)), 'fallback_k': int(fb.get('k', base.fallback_k)), 'fallback_enabled': bool(fb.get('enabled', base.fallback_enabled)), 'summaries': _mode((d.get('summariser') or {}).get('mode', base.summaries)), 'corpus_profile': str(d['_corpus_profile']) if d.get('_corpus_profile') else None}
        config = cls(**kw)
        if config.retrieval_strategy not in {'legacy', 'hierarchical_v1'}:
            raise ValueError('unknown retrieval strategy')
        for key in ('nodes', 'sources', 'view_floor', 'direct'):
            value = getattr(config, f'navigation_{key}')
            if type(value) is not int or value < (1 if key in {'nodes', 'sources'} else 0):
                raise ValueError('invalid navigation budget: ' + key)
        if config.s2_leaf_window not in {'prefix', 'centered'}:
            raise ValueError('config/evoforestmem.yaml: s2.extra_leaf_window must be prefix or centered')
        return config

    @classmethod
    def load(cls, path: str | Path | None=None) -> 'PipelineConfig':
        return cls.from_dict(load_yaml(path))

def _check(block: Mapping[str, Any], known: set[str], where: str) -> None:
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(f'config/evoforestmem.yaml: {where} has no key(s) {unknown}; known keys are {sorted(known)}')
_CORPUS_PROFILE_SET_WHITELIST: Mapping[str, frozenset[str] | None] = {'s1': frozenset({'dense_weight', 'max_hits_per_tree', 'w_collapsed', 'collapsed_recall'}), 'stages': frozenset({'s2', 's3', 's2_source'}), 'expansion': frozenset({'bridge_hop'}), 's4': frozenset({'lifecycle_marks', 'marker_marks', 'leaf_up_headers', 'duration_headers', 'sufficiency_header', 'norm_spans', 'validity_overlay'}), 'validity': frozenset({'penalty'}), 'ff': None}

def corpus_profile_match(profiles: Sequence[Mapping[str, Any]] | None, corpus: str) -> Mapping[str, Any] | None:

    exact: Mapping[str, Any] | None = None
    best_glob: Mapping[str, Any] | None = None
    best_len = -1
    for p in profiles or []:
        m = str(p.get('match') or '')
        if not m:
            continue
        if m == corpus:
            exact = p
            break
        if m.endswith('*') and corpus.startswith(m[:-1]):
            if len(m) > best_len:
                best_glob, best_len = (p, len(m))
    return exact or best_glob

def apply_corpus_profile(d: Mapping[str, Any], corpus: str | None) -> dict[str, Any]:

    hit = corpus_profile_match((d or {}).get('corpus_profiles'), str(corpus) if corpus else '')
    if hit is None:
        return dict(d)
    name = str(hit.get('match'))
    out = dict(d)
    for section, kv in (hit.get('set') or {}).items():
        allowed = _CORPUS_PROFILE_SET_WHITELIST.get(section)
        if section not in _CORPUS_PROFILE_SET_WHITELIST:
            raise ValueError(f'corpus profile {name!r}: unknown section {section!r}; known sections are {sorted(_CORPUS_PROFILE_SET_WHITELIST)}')
        if allowed is not None:
            unknown = sorted(set(kv) - allowed)
            if unknown:
                raise ValueError(f'corpus profile {name!r}: keys not in the whitelist for {section}: {unknown}')
        sec = dict(out.get(section) or {})
        sec.update(kv)
        out[section] = sec
    out['_corpus_profile'] = name
    return out

def _apply(base: Any, over: Mapping[str, Any] | None, where: str, skip: set[str] | None=None) -> Any:

    if not over:
        return base
    skip = skip or set()
    known = {f.name for f in base.__dataclass_fields__.values()}
    unknown = sorted(set(over) - known - skip)
    if unknown:
        raise ValueError(f'config/evoforestmem.yaml: {where} has no field(s) {unknown}; known fields are {sorted(known)}')
    return replace(base, **{k: v for k, v in over.items() if k in known})

def _mode(value: Any) -> str:

    if isinstance(value, bool):
        return 'eager' if value else 'off'
    return str(value)

def load_yaml(path: str | Path | None=None) -> dict[str, Any]:

    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return {}
    import yaml
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}

@dataclass
class PipelineResult:

    context: str
    provenance: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    spec: QuerySpec | None = None
    plan: S1Plan | None = None
    s2: S2Result | None = None
    s3: JoinResult | None = None
    assembled: Assembled | None = None

    @property
    def fallback(self) -> bool:
        return bool(self.stats.get('fallback'))

    @property
    def fallback_reason(self) -> FallbackReason | None:
        return self.stats.get('fallback_reason')
