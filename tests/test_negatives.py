"""Negative phrases: never a target phrase; same-noun off-colors only for nouns told apart by color."""

from segtrack.attributes import build_negatives
from segtrack.parser import parse_task


def negatives_for(text: str) -> set[str]:
    return set(build_negatives(parse_task(text)))


def test_never_contains_target_phrases_or_nouns():
    text = "Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork"
    neg = negatives_for(text)
    assert not neg & {"yellow cup", "white cup", "gray cup", "cup", "spoon", "fork"}
    assert {"bowl", "yellow bowl", "white bowl", "gray bowl", "robot gripper", "container"} <= neg


def test_same_noun_off_color_added_when_noun_is_told_apart_by_color():
    # Bowls come in two task colors (white, yellow); gray is used only for a cup → "gray bowl"
    # is a distractor. Spoon and fork are bare nouns: no colored variants of them.
    neg = negatives_for("Place the white bowl, yellow bowl, white cup, yellow cup, gray cup, spoon, and fork "
                        "into the container.")
    assert "gray bowl" in neg
    assert not {n for n in neg if n.endswith((" spoon", " fork"))}
    assert not neg & {"white cup", "yellow cup", "gray cup", "white bowl", "yellow bowl"}


def test_single_instance_nouns_keep_color_descriptive():
    # One pink fork and one green spoon: a pink fork that reads white must not be rejected, so
    # no "white fork" / "pink spoon" negatives. Bowls (three colors) do get the other colors.
    neg = negatives_for("Place the yellow bowl, white bowl, and gray bowl into the container. "
                        "Then, put the pink fork and green spoon inside the container as well.")
    assert {"pink bowl", "green bowl"} <= neg
    assert not {n for n in neg if n.endswith((" fork", " spoon"))}
