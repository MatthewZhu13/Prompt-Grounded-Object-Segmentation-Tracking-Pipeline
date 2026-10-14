"""Parse a natural-language robot instruction into the set of target objects to track.

    "Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork."
    → targets:      1 yellow cup, 2 white cup, 3 gray cup, 4 spoon, 5 fork
      destinations: container

Approach (rules on top of spaCy part-of-speech tags, no hardcoded object names):

1. Split the instruction into clauses (sentence ends, "then").
2. In each clause, drop the leading verb ("place", "add", "pick up") and split off the
   destination at the first destination preposition ("into the container").
3. Split the remaining object list on commas / conjunctions and parse each item on its own.
   spaCy tags short imperative sentences unreliably (e.g. "fork" as a verb), but tags
   isolated noun phrases like "the fork" reliably, so items are parsed one at a time.
4. Per item: head noun = rightmost noun; attributes = modifiers before it; count from
   numbers ("two forks"), quantifiers ("all cups") or plurality ("cups" → unknown count).
   Bare adjectives share the next item's noun: "blue, white, and pink bowls" → 3 bowls.

Each known-count instance becomes its own Target with a stable ID. Plural nouns without a
number ("cups", "all bowls") become one open-ended Target meaning "every instance".
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

SPACY_MODEL = "en_core_web_sm"

# A noun after one of these is where objects go, not an object to track.
DESTINATION_PREPS = {"into", "onto", "inside", "in", "on", "to", "within", "toward", "towards"}
# Leading words to skip before the verb of a clause.
CLAUSE_LEADERS = {"then", "also", "finally", "next", "please", "and", "now", "first"}
QUANTIFIERS_OPEN = {"all", "every", "each", "some", "any", "remaining", "other"}
QUANTIFIERS_COUNT = {"both": 2}
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
MODIFIER_POS = {"ADJ", "NOUN", "PROPN", "VERB"}   # VERB covers participles: "folded towel"


@dataclass(frozen=True)
class Target:
    id: int                         # stable track ID, 1-based, in order of mention
    phrase: str                     # full description, e.g. "yellow cup" (→ SigLIP)
    noun: str                       # head noun, singular, e.g. "cup" (→ Grounding DINO)
    attributes: tuple[str, ...]     # e.g. ("yellow",)
    open_ended: bool = False        # True: unknown count ("cups") — track every instance
    instance: int = 1               # 1..k when the same phrase is requested k times

    @property
    def label(self) -> str:
        """Display name; disambiguates repeated phrases ("yellow cup #2")."""
        if self.open_ended:
            return f"{self.phrase} (all)"
        return self.phrase if self.instance == 1 else f"{self.phrase} #{self.instance}"


@dataclass
class ParsedTask:
    text: str
    targets: list[Target]
    destinations: list[str] = field(default_factory=list)

    @property
    def nouns(self) -> list[str]:
        """Unique target nouns in order of first mention (the Grounding DINO query)."""
        return list(dict.fromkeys(t.noun for t in self.targets))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "targets": [asdict(t) | {"label": t.label} for t in self.targets],
            "destinations": self.destinations,
        }


@dataclass
class _Mention:
    """One object mention before IDs are assigned."""

    noun: str
    attributes: list[str]
    count: int | None               # None = unknown / every instance

    @property
    def phrase(self) -> str:
        return " ".join([*self.attributes, self.noun])


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load(SPACY_MODEL)



def parse_task(text: str) -> ParsedTask:
    """Extract target objects (with attributes and counts) and destinations from an instruction."""
    mentions: list[_Mention] = []
    destinations: list[str] = []
    for clause in _split_clauses(text):
        clause_mentions, clause_dests = _parse_clause(clause)
        mentions += clause_mentions
        destinations += clause_dests

    targets = _assign_ids(_merge_repeats(mentions))
    if not targets:
        raise ValueError(f"no target objects found in task: {text!r}")
    return ParsedTask(text=text, targets=targets, destinations=list(dict.fromkeys(destinations)))


# ---------- clause level ----------

def _split_clauses(text: str) -> list[str]:
    parts = re.split(r"[.;!?]+|,?\s*\bthen\b\s*,?", text, flags=re.IGNORECASE)
    return [p.strip(" ,") for p in parts if p and p.strip(" ,")]


def _parse_clause(clause: str) -> tuple[list[_Mention], list[str]]:
    """Return (object mentions, destination phrases) for one clause."""
    # Prefixing "you should" makes spaCy read the leading word as a verb; on its own,
    # "Place cups, bowls, ..." tags "Place" as a noun.
    doc = _nlp()("you should " + clause[0].lower() + clause[1:])
    tokens = list(doc)[2:]

    i = 0
    while i < len(tokens) and (tokens[i].lower_ in CLAUSE_LEADERS or tokens[i].is_punct):
        i += 1
    if i < len(tokens) and tokens[i].pos_ in {"VERB", "AUX"}:
        i += 1
        while i < len(tokens) and tokens[i].dep_ == "prt":     # "pick up", "put down"
            i += 1

    objects, dest, rest = [], [], []
    j = i
    while j < len(tokens):
        tok = tokens[j]
        if tok.lower_ in DESTINATION_PREPS and tok.pos_ == "ADP":
            j += 1
            while j < len(tokens) and not (tokens[j].is_punct or tokens[j].pos_ == "CCONJ"):
                dest.append(tokens[j])
                j += 1
            rest = tokens[j + 1:]                               # e.g. ", stacking them neatly"
            break
        objects.append(tok)
        j += 1

    mentions = _parse_object_list(objects)
    dests = [d for d in [_destination_phrase(dest)] if d]
    if rest:                                                    # trailing clause after the destination
        more_m, more_d = _parse_clause(" ".join(t.text for t in rest))
        mentions += more_m
        dests += more_d
    return mentions, dests


def _destination_phrase(tokens) -> str | None:
    words = [t for t in tokens if t.pos_ not in {"DET", "PUNCT", "ADV"}]
    nouns = [k for k, t in enumerate(words) if t.pos_ in {"NOUN", "PROPN"}]
    if not nouns:
        return None
    # "on top of the box" → the last noun is the destination object
    head = nouns[-1]
    start = head
    while start > 0 and words[start - 1].pos_ in MODIFIER_POS:
        start -= 1
    return " ".join([*(w.lower_ for w in words[start:head]), words[head].lemma_.lower()])


# ---------- item level ----------

def _parse_object_list(tokens) -> list[_Mention]:
    """Split an object list on commas / conjunctions and parse each item."""
    items, current = [], []
    for tok in tokens:
        if tok.is_punct or tok.pos_ == "CCONJ" or tok.lower_ in {"plus", "&"}:
            if current:
                items.append(current)
            current = []
        else:
            current.append(tok.text)
    if current:
        items.append(current)

    parsed_items = [_parse_item(" ".join(words)) for words in items]
    # A bare single word before a later modified noun may be a shared-noun adjective that
    # spaCy tagged as a noun in isolation: "green" in "blue, green, and pink bowls".
    for k, (words, parsed) in enumerate(zip(items, parsed_items)):
        later_modified = any(isinstance(p, _Mention) and p.attributes for p in parsed_items[k + 1:])
        if (isinstance(parsed, _Mention) and len(words) == 1 and not parsed.attributes
                and later_modified and _is_predicative_adjective(words[0])):
            parsed_items[k] = [words[0].lower()]

    mentions: list[_Mention] = []
    pending_attrs: list[str] = []       # bare adjectives waiting for a shared noun
    for parsed in parsed_items:
        if parsed is None:
            continue
        if isinstance(parsed, list):    # adjectives only, e.g. "blue" in "blue, white, and pink bowls"
            pending_attrs += parsed
            continue
        if pending_attrs:
            # "blue, white, and pink bowls": the last item's first modifier is its own,
            # any further modifiers ("plastic" in "red and blue plastic cups") are shared.
            own, shared = parsed.attributes[:1], parsed.attributes[1:]
            for attr in [*pending_attrs, *own]:
                mentions.append(_Mention(parsed.noun, [attr, *shared], 1))
            pending_attrs = []
        else:
            mentions.append(parsed)
    return mentions


def _is_predicative_adjective(word: str) -> bool:
    """True if `word` reads as an adjective in "it is <word>" (colors, sizes, materials)."""
    return _nlp()(f"it is {word}")[2].pos_ == "ADJ"


def _parse_item(item: str) -> _Mention | list[str] | None:
    """Parse one list item. Returns a mention, a list of bare adjectives, or None (no object)."""
    doc = _nlp()(item)
    if doc[0].pos_ not in {"DET", "NUM"} and doc[0].lower_ not in QUANTIFIERS_OPEN | set(QUANTIFIERS_COUNT):
        # Isolated noun phrases tag far more reliably with a determiner ("the fork" vs "fork").
        doc = _nlp()("the " + item)
    words = [t for t in doc if not t.is_punct]

    if any(t.pos_ == "PRON" for t in words):
        return None                                             # "them", "it"
    content = [t for t in words if t.pos_ not in {"DET", "NUM"} and t.lower_ not in QUANTIFIERS_OPEN]
    if not content:
        return None
    if all(t.pos_ == "ADJ" for t in content):
        return [t.lower_ for t in content]

    heads = [k for k, t in enumerate(content) if t.pos_ in {"NOUN", "PROPN"}]
    head_idx = heads[-1] if heads else len(content) - 1         # fallback: last content word
    head = content[head_idx]
    attributes = [t.lower_ for t in content[:head_idx] if t.pos_ in MODIFIER_POS]

    count: int | None = 1
    for t in words:
        if t.like_num:
            count = NUMBER_WORDS.get(t.lower_) or (int(t.text) if t.text.isdigit() else None)
        elif t.lower_ in QUANTIFIERS_COUNT:
            count = QUANTIFIERS_COUNT[t.lower_]
        elif t.lower_ in QUANTIFIERS_OPEN:
            count = None
    if count == 1 and head.tag_ in {"NNS", "NNPS"}:
        count = None                                            # bare plural: "cups"

    noun = head.lemma_.lower() if head.pos_ in {"NOUN", "PROPN"} else head.lower_
    return _Mention(noun=noun, attributes=attributes, count=count)


# ---------- IDs ----------

def _merge_repeats(mentions: list[_Mention]) -> list[_Mention]:
    """Combine repeated phrases: "yellow cup ... yellow cup" → count 2."""
    merged: dict[str, _Mention] = {}
    for m in mentions:
        if m.phrase not in merged:
            merged[m.phrase] = _Mention(m.noun, list(m.attributes), m.count)
            continue
        prev = merged[m.phrase]
        prev.count = None if prev.count is None or m.count is None else prev.count + m.count
    return list(merged.values())


def _assign_ids(mentions: list[_Mention]) -> list[Target]:
    targets: list[Target] = []
    for m in mentions:
        if m.count is None:
            targets.append(Target(len(targets) + 1, m.phrase, m.noun, tuple(m.attributes), open_ended=True))
        else:
            for k in range(1, m.count + 1):
                targets.append(Target(len(targets) + 1, m.phrase, m.noun, tuple(m.attributes), instance=k))
    return targets


def _main() -> None:
    p = argparse.ArgumentParser(description="Parse a task instruction into target objects.")
    p.add_argument("task")
    p.add_argument("--json", action="store_true", help="print the full parse as JSON")
    args = p.parse_args()

    parsed = parse_task(args.task)
    if args.json:
        print(json.dumps(parsed.to_dict(), indent=2))
        return
    for t in parsed.targets:
        print(f"  {t.id}: {t.label:24} noun={t.noun!r} attributes={list(t.attributes)}")
    print(f"  destinations: {parsed.destinations}")
    print(f"  detector query: {' . '.join(parsed.nouns)} .")


if __name__ == "__main__":
    _main()
