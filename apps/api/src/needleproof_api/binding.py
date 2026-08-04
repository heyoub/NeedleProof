from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BindingProfile, ObservationKind
from .util import canonical_json, normalize_evidence_text, sha256_text

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
_COPULA = re.compile(r"\s*(?:is|are|was|were)\s*", re.IGNORECASE)
_REPORTED = re.compile(
    r"\s*(?:"
    r"at|closed\s+at|ended(?:\s+the\s+(?:year|quarter|month|period))?\s+at|"
    r"amounted\s+to|reached|reported\s+at|rose\s+to|stood\s+at|"
    r"totaled|totalled|was\s+reported\s+at|were\s+reported\s+at"
    r")\s*",
    re.IGNORECASE,
)
_COLON = re.compile(r"\s*:\s*")
_ANAPHOR = r"(?:it|this|that|the\s+figure)"
_ANAPHORIC_CONNECTOR = (
    r"(?:is|was|remained\s+at|reached|stood\s+at|"
    r"ended(?:\s+the\s+(?:year|quarter|month|period))?\s+at)"
)
_SAME_SENTENCE_BINDING_PATTERN = (
    rf"(?P<prefix>[^,;:.!?\d]*?)\band\s+{_ANAPHOR}\s+{_ANAPHORIC_CONNECTOR}\s*"
)
_NEXT_SENTENCE_BINDING_PATTERN = (
    rf"(?P<prefix>[^.;!?\d]*)\.\s*{_ANAPHOR}\s+{_ANAPHORIC_CONNECTOR}\s*"
)
_FOLLOWING_SENTENCE_ANAPHORIC_PATTERN = (
    rf"(?:(?:by|at|as\s+of|on|for|in|during|through)\b.*?)?"
    rf"{_ANAPHOR}\s+{_ANAPHORIC_CONNECTOR}\s*"
)
_DISALLOWED_PREFIX = re.compile(
    r"\b(?:after|although|and|because|before|but|if|nor|or|so|unless|until|when|"
    r"whereas|while|yet)\b",
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

BINDING_CONTRACT_SPEC = {
    "version": "positive-bindings-v2-atomic-assertions",
    "profiles": [profile.value for profile in BindingProfile],
    "authorized_observation_kinds": sorted(kind.value for kind in AUTHORIZED_OBSERVATION_KINDS),
    "copula_pattern": _COPULA.pattern,
    "reported_pattern": _REPORTED.pattern,
    "colon_pattern": _COLON.pattern,
    "anaphor_pattern": _ANAPHOR,
    "anaphoric_connector_pattern": _ANAPHORIC_CONNECTOR,
    "same_sentence_binding_pattern": _SAME_SENTENCE_BINDING_PATTERN,
    "next_sentence_binding_pattern": _NEXT_SENTENCE_BINDING_PATTERN,
    "following_sentence_anaphoric_pattern": _FOLLOWING_SENTENCE_ANAPHORIC_PATTERN,
    "disallowed_prefix_pattern": _DISALLOWED_PREFIX.pattern,
    "before_metric_temporal_gap_pattern": _BEFORE_METRIC_TEMPORAL_GAP.pattern,
    "after_value_temporal_gap_pattern": _AFTER_VALUE_TEMPORAL_GAP.pattern,
    "post_value_tail": (
        "only terminal punctuation after the value or its bound trailing temporal anchor"
    ),
    "unresolved_predicate_detection": "known positive connector with nonempty predicate",
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


def word_phrase_spans(needle: str, haystack: str) -> list[Span]:
    expected = _WORD.findall(needle.casefold())
    observed = list(_WORD.finditer(haystack.casefold()))
    if not expected:
        return []
    width = len(expected)
    return [
        (observed[index].start(), observed[index + width - 1].end())
        for index in range(len(observed) - width + 1)
        if [match.group() for match in observed[index : index + width]] == expected
    ]


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
    spans = word_phrase_spans(value, assertion)
    return spans, bool(spans)


def numeric_value_candidates(text: str) -> tuple[tuple[str, Span], ...]:
    """Return source numeric measurements and spans without doing arithmetic."""

    return tuple(
        (match.group().strip(), (match.start(), match.end())) for match in _NUMERIC.finditer(text)
    )


def has_unresolved_metric_predicate(metric_anchor: str, assertion: str) -> bool:
    """Detect a known positive predicate shape whose qualitative value is unresolved.

    This deliberately reuses only the closed binding connectors. A bare mention such as
    ``Total headcount - source quote`` is not evidence, while ``Credit rating was stable``
    must prevent an authoritative absence conclusion even though no numeric candidate exists.
    """

    normalized_assertion = normalize_evidence_text(assertion)[0].casefold()
    normalized_metric = normalize_evidence_text(metric_anchor)[0].casefold()
    for _start, end in word_phrase_spans(normalized_metric, normalized_assertion):
        suffix = normalized_assertion[end:]
        for connector in (_COPULA, _REPORTED, _COLON):
            match = connector.match(suffix)
            if match and _WORD.search(suffix[match.end() :]):
                return True
    return False


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
    if same_sentence and not _DISALLOWED_PREFIX.search(same_sentence.group("prefix")):
        return BindingProfile.SAME_SENTENCE_ANAPHORIC

    next_sentence = re.fullmatch(
        _NEXT_SENTENCE_BINDING_PATTERN,
        between,
        flags=re.IGNORECASE,
    )
    if next_sentence and not _DISALLOWED_PREFIX.search(next_sentence.group("prefix")):
        return BindingProfile.NEXT_SENTENCE_ANAPHORIC
    return None


def _next_sentence_anaphoric_profile(
    assertion: str,
    metric_span: Span,
    value_span: Span,
) -> bool:
    metric_sentence_end_match = re.search(r"[.!?]\s+", assertion[metric_span[1] :])
    if metric_sentence_end_match is None:
        return False
    boundary_start = metric_span[1] + metric_sentence_end_match.start()
    next_start = metric_span[1] + metric_sentence_end_match.end()
    if not (boundary_start < value_span[0]):
        return False
    candidate = assertion[next_start : value_span[0]]
    return bool(
        re.fullmatch(
            _FOLLOWING_SENTENCE_ANAPHORIC_PATTERN,
            candidate,
            flags=re.IGNORECASE,
        )
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
    for span in word_phrase_spans(temporal_anchor, assertion):
        if metric_span[1] <= span[0] and span[1] <= value_span[0]:
            if word_phrase_spans(temporal_anchor, between):
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


def bind_observation(
    *,
    metric_anchor: str,
    value_text: str,
    kind: ObservationKind,
    temporal_anchor: str | None,
    assertion: str,
) -> BindingResult:
    normalized_assertion = normalize_evidence_text(assertion)[0].casefold()
    normalized_metric = normalize_evidence_text(metric_anchor)[0].casefold()
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
    if kind not in AUTHORIZED_OBSERVATION_KINDS or not _kind_accepts_signature(
        kind, normalized_value
    ):
        return BindingResult(None, True, "value_role_not_authorized")

    for metric_span in metric_spans:
        for value_span in value_spans:
            if value_span[0] < metric_span[1]:
                continue
            between = normalized_assertion[metric_span[1] : value_span[0]]
            profile = _profile_for_between(between)
            if profile is None and _next_sentence_anaphoric_profile(
                normalized_assertion, metric_span, value_span
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
    return BindingResult(None, True, "no_positive_binding_profile")
