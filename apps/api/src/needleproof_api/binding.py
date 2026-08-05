from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from .models import BindingProfile, ObservationKind
from .util import canonical_json, metric_tokens, normalize_evidence_text, sha256_text

NumericSignature = tuple[str, str, str, str]
Span = tuple[int, int]

_NUMERIC = re.compile(
    r"(?P<open>\()?\s*(?P<sign_before>[+-])?\s*(?P<currency>[$£€])?\s*"
    r"(?P<sign_after>[+-])?\s*(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>billion|million|thousand|basis\s+points?|bps|percent|%|per\s+share)?"
    r"\s*(?P<close>\))?",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W_]+")
_INTRA_PHRASE_FORMATTING = re.compile(r"[\s/&+'\"()\[\]{}\-‐‑‒–—]+")
_COPULA = re.compile(r"\s*(?:is|are|was|were)\b\s*", re.IGNORECASE)
_REPORTED = re.compile(
    r"\s*(?:"
    r"at|closed\s+at|ended(?:\s+the\s+(?:year|quarter|month|period))?\s+at|"
    r"amounted\s+to|reached|reported\s+at|rose\s+to|stood\s+at|"
    r"totaled|totalled|was\s+reported\s+at|were\s+reported\s+at"
    r")\b\s*",
    re.IGNORECASE,
)
_COLON = re.compile(r"\s*:\s*")
_QUALITATIVE_STATE = re.compile(
    r"\s*(?:remained(?:\s+at)?|reported|stayed(?:\s+at)?)\b\s*",
    re.IGNORECASE,
)
_PREDICATE_QUALIFIER_GAP = re.compile(
    r"\s*(?:[^\W\d_]+\s*){1,8}",
    re.IGNORECASE,
)
_ANAPHOR = r"(?:it|this|that|the\s+figure)"
_ANAPHORIC_CONNECTOR = (
    r"(?:is|was|remained\s+at|reached|stood\s+at|"
    r"ended(?:\s+the\s+(?:year|quarter|month|period))?\s+at)"
)
_POSITIVE_ANAPHORIC_STATE = (
    r"(?:is|are|was|were)\s+(?:available|disclosed|reported|stable|stated|unchanged)"
)
_PRE_METRIC_QUALIFIER_TOKEN = r"(?:[^\W\d_]+|(?:[^\W\d_]\.){2,})"
_PRE_METRIC_QUALITATIVE_STATE = re.compile(
    r"\b(?:available|disclosed|flat|reported|stable|stated|unchanged)"
    rf"(?:\s+{_PRE_METRIC_QUALIFIER_TOKEN}){{0,3}}\s+$",
    re.IGNORECASE,
)
_POSITIVE_ANAPHORIC_DESCRIPTOR = (
    r"(?:is|are|was|were)\s+the\s+(?:figure|metric|number)\s+that\s+"
    r"(?:actually\s+)?matters(?:\s+for\s+[^\W\d_]+){0,4}"
)
_POSITIVE_ANAPHORIC_ANTECEDENT = (
    rf"(?:{_POSITIVE_ANAPHORIC_STATE}|{_POSITIVE_ANAPHORIC_DESCRIPTOR})"
)
_SAME_SENTENCE_BINDING_PATTERN = (
    rf"\s*{_POSITIVE_ANAPHORIC_ANTECEDENT}\s+"
    rf"and\s+{_ANAPHOR}\s+{_ANAPHORIC_CONNECTOR}\s*"
)
_NEXT_SENTENCE_BINDING_PATTERN = (
    rf"\s*{_POSITIVE_ANAPHORIC_ANTECEDENT}\s*\.\s*"
    rf"{_ANAPHOR}\s+{_ANAPHORIC_CONNECTOR}\s*"
)
_IMMEDIATE_FOLLOWING_ANAPHORIC_PATTERN = (
    rf"\s*(?:(?P<lead>(?:by|at|as\s+of|on|for|in|during|through)\b[^.!?]*?)\s+)?"
    rf"{_ANAPHOR}\s+{_ANAPHORIC_CONNECTOR}\s*"
)
_ANAPHORIC_NUMERIC_TEMPORAL_TAIL = re.compile(
    r"\s*(?:(?:as\s+of|at|during|for|in|on|through)\s+[^.!?]+)?\s*",
    re.IGNORECASE,
)
_ANAPHORIC_TEMPORAL_LEAD_PREFIX = re.compile(
    r"\s*(?:by|at|as\s+of|on|for|in|during|through)\s+"
    r"(?:(?:the|fiscal|calendar|year|quarter|month|period|end|ending|ended|as|of|on)\s+)*",
    re.IGNORECASE,
)
_AFTER_VALUE_TEMPORAL_GAP = re.compile(
    r"\s*,?\s*(?:(?:as\s+of|at|during|for|in|on|through)\s+)?$",
    re.IGNORECASE,
)
_BEFORE_METRIC_TEMPORAL_GAP = re.compile(
    r"\s*(?:,|;)?\s*(?:(?:the\s+)?(?:figure|metric)\s+)?(?:has|reported|showed|listed)?\s*$",
    re.IGNORECASE,
)
_TERMINAL_ASSERTION_TAIL = re.compile(r"\s*[.!?]?\s*")
_DIRECT_METRIC_PREFIX = re.compile(r"\s*(?:the\s+)?", re.IGNORECASE)
_TEMPORAL_METRIC_PREFIX = re.compile(
    r"\s*(?:(?:as\s+of|at|by|during|for|in|on|through)\s+)?(?:the\s+)?",
    re.IGNORECASE,
)
_MONTH_NAME = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
)
_CALENDAR_DATE = (
    rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|"
    rf"\d{{1,2}}\s+{_MONTH_NAME}\s+\d{{4}}|"
    rf"{_MONTH_NAME}\s+\d{{1,2}},?\s+\d{{4}})"
)
_DATED_PERIOD = (
    rf"(?:the\s+)?(?:(?:fiscal|calendar)\s+)?(?:year|quarter|month|period)\s+"
    rf"(?:end|ended|ending)(?:\s+on)?\s+{_CALENDAR_DATE}"
)
_NAMED_PERIOD = (
    rf"(?:"
    rf"{_DATED_PERIOD}|"
    rf"(?:fy|fiscal\s+year|calendar\s+year|year|q[1-4]|"
    rf"first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter)"
    rf"(?:\s+\d{{4}})?(?:\s+(?:end|ended|ending|start|started|beginning))?|"
    rf"(?:end|ended|ending|start|started|beginning)\s+(?:of\s+)?(?:the\s+)?"
    rf"(?:(?:fiscal|calendar)\s+)?(?:year|quarter|month|period)|"
    rf"{_MONTH_NAME}(?:\s+\d{{4}})?(?:\s+(?:call|quarter|month|period|year(?:\s+end)?))?"
    rf")"
)
_AUTHORIZED_TEMPORAL_ANCHOR = re.compile(
    rf"\s*(?:(?:as\s+of|at|by|during|for|in|on|through)\s+)?"
    rf"(?:{_CALENDAR_DATE}|(?:19|20)\d{{2}}|{_NAMED_PERIOD})\s*",
    re.IGNORECASE,
)

AUTHORIZED_OBSERVATION_KINDS = frozenset(
    {
        ObservationKind.REPORTED_LEVEL,
        ObservationKind.REPORTED_RATE,
        ObservationKind.REPORTED_PER_SHARE,
    }
)
_UNIT_ALIASES = {
    "bps": "basis points",
    "basis point": "basis points",
    "%": "percent",
}
_RATE_UNITS = frozenset({"basis point", "basis points", "bps", "percent", "%"})
_PER_SHARE_UNITS = frozenset({"per share"})
_EXPLICIT_QUALITATIVE_NEGATION = re.compile(
    r"(?:not|never|no\s+longer|neither|nor)\b",
    re.IGNORECASE,
)

BINDING_CONTRACT_SPEC = {
    "version": "positive-bindings-v18-source-only-compact-disambiguation",
    "profiles": [profile.value for profile in BindingProfile],
    "authorized_observation_kinds": sorted(kind.value for kind in AUTHORIZED_OBSERVATION_KINDS),
    "copula_pattern": _COPULA.pattern,
    "reported_pattern": _REPORTED.pattern,
    "colon_pattern": _COLON.pattern,
    "anaphor_pattern": _ANAPHOR,
    "anaphoric_connector_pattern": _ANAPHORIC_CONNECTOR,
    "positive_anaphoric_antecedent_pattern": _POSITIVE_ANAPHORIC_ANTECEDENT,
    "same_sentence_binding_pattern": _SAME_SENTENCE_BINDING_PATTERN,
    "next_sentence_binding_pattern": _NEXT_SENTENCE_BINDING_PATTERN,
    "next_sentence_boundary": "period_only",
    "numeric_anaphoric_antecedent": (
        "must_equal_selected_canonical_numeric_signature_unless_a_new_temporal_lead_is_bound"
    ),
    "immediate_following_anaphoric_pattern": _IMMEDIATE_FOLLOWING_ANAPHORIC_PATTERN,
    "separate_numeric_followup": (
        "anaphoric_connector_then_immediate_measurement_then_optional_temporal_tail"
    ),
    "anaphoric_numeric_temporal_tail_pattern": _ANAPHORIC_NUMERIC_TEMPORAL_TAIL.pattern,
    "anaphoric_temporal_lead_prefix_pattern": _ANAPHORIC_TEMPORAL_LEAD_PREFIX.pattern,
    "before_metric_temporal_gap_pattern": _BEFORE_METRIC_TEMPORAL_GAP.pattern,
    "after_value_temporal_gap_pattern": _AFTER_VALUE_TEMPORAL_GAP.pattern,
    "post_value_tail": (
        "only terminal punctuation after the value or its bound trailing temporal anchor"
    ),
    "direct_metric_prefix_pattern": _DIRECT_METRIC_PREFIX.pattern,
    "temporal_metric_prefix_pattern": _TEMPORAL_METRIC_PREFIX.pattern,
    "authorized_temporal_anchor_pattern": _AUTHORIZED_TEMPORAL_ANCHOR.pattern,
    "pre_metric_subject": "complete metric or bound leading temporal anchor",
    "metric_phrase_separator_policy": (
        "shared_span_preserving_kind_aware_tokens_with_2_to_3_letter_compact_dotted_"
        "initialism_equivalence_and_longer_uppercase_word_typography"
    ),
    "bare_compact_all_caps_policy": (
        "reject_single_compact_metric_in_all_caps_assertion_unless_source_is_dotted_or_metric_"
        "phrase_has_multiple_tokens"
    ),
    "qualitative_value_identity": "normalized_casefolded_word_token_sequence",
    "qualitative_value_token_kind": "ignored_while_metric_token_kind_remains_authoritative",
    "temporal_anchor_identity": "normalized_casefolded_word_token_sequence",
    "temporal_anchor_token_kind": "ignored_while_metric_token_kind_remains_authoritative",
    "explicit_qualitative_negation_pattern": _EXPLICIT_QUALITATIVE_NEGATION.pattern,
    "explicit_qualitative_negation": "reject_before_positive_profile_matching",
    "intra_phrase_formatting_pattern": _INTRA_PHRASE_FORMATTING.pattern,
    "unresolved_predicate_detection": (
        "known positive connector after at most eight punctuation-free qualifier words "
        "with a nonempty predicate"
    ),
    "unresolved_qualitative_state_pattern": _QUALITATIVE_STATE.pattern,
    "pre_metric_qualitative_state_pattern": _PRE_METRIC_QUALITATIVE_STATE.pattern,
    "pre_metric_qualifier_token_pattern": _PRE_METRIC_QUALIFIER_TOKEN,
    "pre_metric_qualifier_limit": 3,
    "predicate_qualifier_gap_pattern": _PREDICATE_QUALIFIER_GAP.pattern,
    "unknown_syntax": "reject",
    "compound_numeric_observation": "reject",
}
BINDING_CONTRACT_SHA256 = sha256_text(canonical_json(BINDING_CONTRACT_SPEC))
NUMERIC_CONTRACT_SPEC = {
    "version": "fixed-point-source-digits-v1",
    "numeric_pattern": _NUMERIC.pattern,
    "canonicalization": "strip_integer_leading_zeros_and_fractional_trailing_zeros",
    "unit_aliases": _UNIT_ALIASES,
    "unit_conversion": "forbidden",
    "arithmetic": "none",
}
NUMERIC_CONTRACT_SHA256 = sha256_text(canonical_json(NUMERIC_CONTRACT_SPEC))


@dataclass(frozen=True, slots=True)
class BindingMatch:
    profile: BindingProfile
    value_role: ObservationKind
    sentence_span: Span
    assertion_span: Span
    metric_span: Span
    value_spans: tuple[Span, ...]
    temporal_span: Span | None


@dataclass(frozen=True, slots=True)
class BindingResult:
    match: BindingMatch | None
    value_text_found: bool
    failure_reason: str | None


def _numeric_signature(match: re.Match[str]) -> NumericSignature:
    parenthesized = bool(match.group("open") and match.group("close"))
    explicit_sign = match.group("sign_before") or match.group("sign_after")
    sign = "-" if parenthesized or explicit_sign == "-" else "+"
    currency = match.group("currency") or ""
    number = match.group("number").replace(",", "")
    unit = re.sub(r"\s+", " ", (match.group("unit") or "").lower())
    return sign, currency, number, unit


def numeric_signature_sequence(text: str) -> tuple[NumericSignature, ...]:
    return tuple(_numeric_signature(match) for match in _NUMERIC.finditer(text))


def numeric_signatures(text: str) -> set[NumericSignature]:
    return set(numeric_signature_sequence(text))


def canonical_decimal_digits(number: str) -> str:
    """Canonicalize fixed-point source digits without Decimal context or rounding."""

    integer, separator, fraction = number.partition(".")
    integer = integer.lstrip("0") or "0"
    if not separator:
        return integer
    fraction = fraction.rstrip("0")
    return f"{integer}.{fraction}" if fraction else integer


def canonical_numeric_signature(signature: NumericSignature) -> NumericSignature:
    sign, currency, number, unit = signature
    canonical_unit = _UNIT_ALIASES.get(unit, unit)
    return sign, currency, canonical_decimal_digits(number), canonical_unit


def canonical_word_phrase(value: str) -> str:
    """Return the binder's punctuation-independent word-token identity."""

    normalized = normalize_evidence_text(value)[0].casefold()
    return " ".join(match.group() for match in _WORD.finditer(normalized))


def word_phrase_spans(
    needle: str,
    haystack: str,
    *,
    preserve_token_kind: bool = True,
) -> list[Span]:
    # Callers may supply a raw PDF metric anchor; the corpus-side haystack is
    # already normalized and its offsets must remain in that source space.
    expected = metric_tokens(normalize_evidence_text(needle)[0])
    observed = metric_tokens(haystack)
    if not expected:
        return []
    width = len(expected)
    spans = []
    for index in range(len(observed) - width + 1):
        window = observed[index : index + width]
        expected_identity = (
            [token.identity for token in expected]
            if preserve_token_kind
            else [token.value for token in expected]
        )
        observed_identity = (
            [token.identity for token in window]
            if preserve_token_kind
            else [token.value for token in window]
        )
        if observed_identity != expected_identity:
            continue
        if all(
            _INTRA_PHRASE_FORMATTING.fullmatch(haystack[left.end : right.start])
            for left, right in pairwise(window)
        ):
            spans.append((window[0].start, window[-1].end))
    return spans


def _value_spans(value: str, assertion: str) -> tuple[list[Span], bool]:
    expected = numeric_signature_sequence(value)
    if expected:
        if len(expected) != 1:
            return [], bool(numeric_signature_sequence(assertion))
        spans = [
            (match.start(), match.end())
            for match in _NUMERIC.finditer(assertion)
            if _numeric_signature(match) == expected[0]
        ]
        return spans, bool(spans)
    # Qualitative values are documented as normalized case-folded word
    # sequences. Token kind is an authority boundary for metric identity, not
    # for values such as the credit rating ``AA``. Keeping those laws separate
    # prevents a case-styled qualitative value from becoming an accidental
    # false negative without reopening ``IT`` == ``It`` metric matching.
    spans = word_phrase_spans(value, assertion, preserve_token_kind=False)
    return spans, bool(spans)


def numeric_value_candidates(text: str) -> tuple[tuple[str, Span], ...]:
    """Return source numeric measurements and spans without doing arithmetic."""

    return tuple(
        (match.group().strip(), (match.start(), match.end())) for match in _NUMERIC.finditer(text)
    )


def has_positive_anaphoric_numeric_followup(text: str) -> bool:
    """Prove a closed numeric follow-up instead of accepting any later digits."""

    normalized = normalize_evidence_text(text)[0].casefold()
    relationship = re.match(_IMMEDIATE_FOLLOWING_ANAPHORIC_PATTERN, normalized, re.IGNORECASE)
    if relationship is None:
        return False
    candidates = numeric_value_candidates(normalized)
    if not candidates:
        return False
    _value, span = candidates[0]
    if normalized[relationship.end() : span[0]].strip():
        return False
    tail = normalized[span[1] :].rstrip(".!? ")
    return bool(_ANAPHORIC_NUMERIC_TEMPORAL_TAIL.fullmatch(tail))


def has_unresolved_metric_predicate(metric_anchor: str, assertion: str) -> bool:
    """Detect a known positive predicate shape whose qualitative value is unresolved.

    This deliberately reuses only the closed binding connectors. A bare mention such as
    ``Total headcount - source quote`` is not evidence, while ``Credit rating was stable``
    must prevent an authoritative absence conclusion even though no numeric candidate exists.
    """

    # Preserve source orthography until metric-token classification. Case-folding
    # here would collapse ``IT`` into the ordinary pronoun ``It`` before the
    # kind-preserving identity law can run.
    normalized_assertion = normalize_evidence_text(assertion)[0]
    normalized_metric = normalize_evidence_text(metric_anchor)[0]
    for metric_start, end in word_phrase_spans(normalized_metric, normalized_assertion):
        if _PRE_METRIC_QUALITATIVE_STATE.search(normalized_assertion[:metric_start]):
            return True
        suffix = normalized_assertion[end:]
        for connector in (_COPULA, _REPORTED, _COLON, _QUALITATIVE_STATE):
            for match in connector.finditer(suffix):
                gap = suffix[: match.start()]
                if gap and not _PREDICATE_QUALIFIER_GAP.fullmatch(gap):
                    continue
                if _WORD.search(suffix[match.end() :]):
                    return True
    return False


def is_authorized_temporal_anchor(value: str | None) -> bool:
    """Accept only closed date, period, and named-event temporal expressions."""

    if value is None:
        return True
    normalized = normalize_evidence_text(value)[0].casefold()
    return bool(_AUTHORIZED_TEMPORAL_ANCHOR.fullmatch(normalized))


def _profile_for_between(between: str) -> BindingProfile | None:
    if _COLON.fullmatch(between):
        return BindingProfile.COLON
    if _COPULA.fullmatch(between):
        return BindingProfile.DIRECT_COPULA
    if _REPORTED.fullmatch(between):
        return BindingProfile.DIRECT_REPORTED

    same_sentence = re.fullmatch(
        _SAME_SENTENCE_BINDING_PATTERN,
        between,
        flags=re.IGNORECASE,
    )
    if same_sentence:
        return BindingProfile.SAME_SENTENCE_ANAPHORIC

    next_sentence = re.fullmatch(
        _NEXT_SENTENCE_BINDING_PATTERN,
        between,
        flags=re.IGNORECASE,
    )
    if next_sentence:
        return BindingProfile.NEXT_SENTENCE_ANAPHORIC
    return None


def _positive_anaphoric_antecedent(
    predicate: str,
    selected_value: str,
    *,
    allow_temporal_variant: bool,
) -> bool:
    if re.fullmatch(_POSITIVE_ANAPHORIC_ANTECEDENT, predicate, flags=re.IGNORECASE):
        return True
    for connector in (_COPULA, _REPORTED, _COLON):
        match = connector.match(predicate)
        if match is None:
            continue
        numeric = _NUMERIC.match(predicate, match.end())
        if numeric is None or not _ANAPHORIC_NUMERIC_TEMPORAL_TAIL.fullmatch(
            predicate[numeric.end() :]
        ):
            continue
        selected = numeric_signature_sequence(selected_value)
        return bool(
            allow_temporal_variant
            or (
                len(selected) == 1
                and canonical_numeric_signature(_numeric_signature(numeric))
                == canonical_numeric_signature(selected[0])
            )
        )
    return False


def _lead_binds_temporal_anchor(lead: str, temporal_anchor: str | None) -> bool:
    if temporal_anchor is None:
        return False
    for start, end in word_phrase_spans(
        temporal_anchor,
        lead,
        preserve_token_kind=False,
    ):
        if _ANAPHORIC_TEMPORAL_LEAD_PREFIX.fullmatch(lead[:start]) and re.fullmatch(
            r"\s*,?\s*", lead[end:]
        ):
            return True
    return False


def _immediate_next_sentence_anaphoric_profile(
    assertion: str,
    metric_span: Span,
    value_span: Span,
    temporal_anchor: str | None,
) -> bool:
    """Prove one positive antecedent and one immediately following pronoun sentence."""

    boundary = re.search(r"\.\s+", assertion[metric_span[1] :])
    if boundary is None:
        return False
    boundary_start = metric_span[1] + boundary.start()
    next_start = metric_span[1] + boundary.end()
    if not (boundary_start < value_span[0]):
        return False
    antecedent = assertion[metric_span[1] : boundary_start]
    candidate = assertion[next_start : value_span[0]]
    match = re.fullmatch(
        _IMMEDIATE_FOLLOWING_ANAPHORIC_PATTERN,
        candidate,
        flags=re.IGNORECASE,
    )
    if match is None:
        return False
    lead = match.group("lead")
    temporal_variant = bool(lead and _lead_binds_temporal_anchor(lead, temporal_anchor))
    if lead and not temporal_variant:
        return False
    selected_value = assertion[value_span[0] : value_span[1]]
    return _positive_anaphoric_antecedent(
        antecedent,
        selected_value,
        allow_temporal_variant=temporal_variant,
    )


def _kind_accepts_signature(kind: ObservationKind, value: str) -> bool:
    signatures = numeric_signature_sequence(value)
    if len(signatures) > 1:
        return False
    unit = signatures[0][3] if signatures else ""
    if kind == ObservationKind.REPORTED_RATE:
        return unit in _RATE_UNITS
    if kind == ObservationKind.REPORTED_PER_SHARE:
        return unit in _PER_SHARE_UNITS
    if kind == ObservationKind.REPORTED_LEVEL:
        return unit not in (_RATE_UNITS | _PER_SHARE_UNITS)
    return False


def _temporal_span(
    assertion: str,
    temporal_anchor: str | None,
    metric_span: Span,
    value_span: Span,
    between: str,
) -> Span | None:
    if temporal_anchor is None:
        return None
    for span in word_phrase_spans(
        temporal_anchor,
        assertion,
        preserve_token_kind=False,
    ):
        if metric_span[1] <= span[0] and span[1] <= value_span[0]:
            if word_phrase_spans(
                temporal_anchor,
                between,
                preserve_token_kind=False,
            ):
                return span
        elif span[1] <= metric_span[0]:
            gap = assertion[span[1] : metric_span[0]]
            if _BEFORE_METRIC_TEMPORAL_GAP.fullmatch(gap):
                return span
        elif value_span[1] <= span[0]:
            gap = assertion[value_span[1] : span[0]]
            if _AFTER_VALUE_TEMPORAL_GAP.fullmatch(gap):
                return span
    return None


def _post_value_tail_is_authorized(
    assertion: str,
    value_span: Span,
    temporal_span: Span | None,
) -> bool:
    """Require the selected assertion to end with the proved value relationship.

    A trailing temporal anchor may complete that relationship. Any other suffix is
    unknown syntax and fails closed, including targets, forecasts, bounds, components,
    deltas, and negations written after an otherwise valid metric/value prefix.
    """

    relationship_end = value_span[1]
    if temporal_span is not None and temporal_span[0] >= value_span[1]:
        relationship_end = temporal_span[1]
    return bool(_TERMINAL_ASSERTION_TAIL.fullmatch(assertion[relationship_end:]))


def _metric_prefix_is_authorized(
    assertion: str,
    metric_span: Span,
    temporal_span: Span | None,
) -> bool:
    """Require the selected metric to own the complete assertion subject.

    A bound leading temporal anchor may precede the metric. Otherwise only an
    optional article is allowed; role, scope, modality, and negation modifiers must
    be included in the canonical metric anchor or the observation fails closed.
    """

    if temporal_span is not None and temporal_span[1] <= metric_span[0]:
        return bool(_TEMPORAL_METRIC_PREFIX.fullmatch(assertion[: temporal_span[0]]))
    return bool(_DIRECT_METRIC_PREFIX.fullmatch(assertion[: metric_span[0]]))


def _bare_compact_metric_is_ambiguous_all_caps(
    metric: str,
    assertion: str,
    metric_span: Span,
) -> bool:
    """Reject a bare compact token whose all-caps source cannot disambiguate pronoun typography."""

    metric_lexemes = metric_tokens(metric)
    if (
        len(metric_lexemes) != 1
        or not metric_lexemes[0].is_initialism
        or "." in assertion[metric_span[0] : metric_span[1]]
    ):
        return False
    cased_characters = [character for character in assertion if character.isalpha()]
    return bool(cased_characters) and all(
        character == character.upper() for character in cased_characters
    )


def bind_observation(
    *,
    metric_anchor: str,
    value_text: str,
    kind: ObservationKind,
    temporal_anchor: str | None,
    assertion: str,
) -> BindingResult:
    # Metric identity is case-insensitive *within* a token kind. Preserve the
    # source spelling long enough to distinguish compact initialisms from words.
    normalized_assertion = normalize_evidence_text(assertion)[0]
    normalized_metric = normalize_evidence_text(metric_anchor)[0]
    normalized_value = normalize_evidence_text(value_text)[0].casefold()
    normalized_temporal = (
        normalize_evidence_text(temporal_anchor)[0].casefold() if temporal_anchor else None
    )

    metric_spans = word_phrase_spans(normalized_metric, normalized_assertion)
    value_spans, value_text_found = _value_spans(normalized_value, normalized_assertion)
    if not metric_spans:
        return BindingResult(None, value_text_found, "metric_anchor_not_in_assertion")
    if not value_text_found:
        return BindingResult(None, False, "value_text_not_in_assertion")
    if normalized_temporal and not is_authorized_temporal_anchor(normalized_temporal):
        return BindingResult(None, True, "temporal_anchor_not_authorized")
    if not numeric_signature_sequence(normalized_value) and _EXPLICIT_QUALITATIVE_NEGATION.search(
        normalized_value
    ):
        return BindingResult(None, True, "negated_qualitative_value")
    if kind not in AUTHORIZED_OBSERVATION_KINDS or not _kind_accepts_signature(
        kind, normalized_value
    ):
        return BindingResult(None, True, "value_role_not_authorized")

    ambiguous_compact_metric = False
    for metric_span in metric_spans:
        if _bare_compact_metric_is_ambiguous_all_caps(
            normalized_metric,
            normalized_assertion,
            metric_span,
        ):
            ambiguous_compact_metric = True
            continue
        for value_span in value_spans:
            if value_span[0] < metric_span[1]:
                continue
            between = normalized_assertion[metric_span[1] : value_span[0]]
            profile = _profile_for_between(between)
            if profile is None and _immediate_next_sentence_anaphoric_profile(
                normalized_assertion,
                metric_span,
                value_span,
                normalized_temporal,
            ):
                profile = BindingProfile.NEXT_SENTENCE_ANAPHORIC
            if profile is None:
                continue
            temporal_span = _temporal_span(
                normalized_assertion,
                normalized_temporal,
                metric_span,
                value_span,
                between,
            )
            if normalized_temporal and temporal_span is None:
                continue
            if not _post_value_tail_is_authorized(
                normalized_assertion,
                value_span,
                temporal_span,
            ):
                continue
            if not _metric_prefix_is_authorized(
                normalized_assertion,
                metric_span,
                temporal_span,
            ):
                continue
            if normalized_temporal and temporal_span and temporal_span[1] <= metric_span[0]:
                profile = BindingProfile.DATED_DIRECT
            return BindingResult(
                BindingMatch(
                    profile=profile,
                    value_role=kind,
                    sentence_span=(0, len(normalized_assertion)),
                    assertion_span=(0, len(normalized_assertion)),
                    metric_span=metric_span,
                    value_spans=(value_span,),
                    temporal_span=temporal_span,
                ),
                True,
                None,
            )
    return BindingResult(
        None,
        True,
        (
            "ambiguous_bare_compact_metric_in_all_caps_assertion"
            if ambiguous_compact_metric
            else "no_positive_binding_profile"
        ),
    )
