"""Parser tests. Most cases are the dataset's own per-episode instructions
(meta/tasks_annotated.parquet); the rest are unseen phrasings to check generalization."""

import pytest

from segtrack.parser import parse_task


def labels(task: str) -> list[str]:
    return [t.label for t in parse_task(task).targets]


@pytest.mark.parametrize("task, expected", [
    # ep 0 — the demo task
    ("Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork.",
     ["yellow cup", "white cup", "gray cup", "spoon", "fork"]),
    # ep 5 — cups and bowls sharing colors
    ("Place the white bowl, yellow bowl, white cup, yellow cup, gray cup, spoon, and fork into the container.",
     ["white bowl", "yellow bowl", "white cup", "yellow cup", "gray cup", "spoon", "fork"]),
    # ep 24 — two sentences, "Then,", attributes on utensils, "inside ... as well"
    ("Place the yellow bowl, white bowl, and gray bowl into the container. "
     "Then, put the pink fork and green spoon inside the container as well.",
     ["yellow bowl", "white bowl", "gray bowl", "pink fork", "green spoon"]),
    # ep 34 — colors sharing one plural noun ("green" is tagged a noun in isolation)
    ("Place blue, white, yellow, green, and pink bowls into the container.",
     ["blue bowl", "white bowl", "yellow bowl", "green bowl", "pink bowl"]),
    # ep 57 — shared nouns in both clauses, ", then"
    ("Place white, yellow, blue, and pink cups into the container, then add blue and white bowls.",
     ["white cup", "yellow cup", "blue cup", "pink cup", "blue bowl", "white bowl"]),
    # ep 53 — repeated phrase means two instances
    ("Place yellow cup, white cup, pink cup, blue cup, and yellow cup into the container. "
     "Then, place blue bowl and white bowl into the container.",
     ["yellow cup", "yellow cup #2", "white cup", "pink cup", "blue cup", "blue bowl", "white bowl"]),
    # ep 4 — no determiners, bare plural (unknown count), lowercase "then"
    ("Place yellow bowl, white bowl, and cups into container, then add spoons and fork.",
     ["yellow bowl", "white bowl", "cup (all)", "spoon (all)", "fork"]),
    # ep 2 — bare plurals right after the verb
    ("Place cups, bowls, and utensils into the container.",
     ["cup (all)", "bowl (all)", "utensil (all)"]),
    # ep 39 — trailing clause after the destination contributes nothing
    ("Place bowls and cups into the container, stacking them neatly.",
     ["bowl (all)", "cup (all)"]),
    # unseen phrasings
    ("Pick up the red block and put it in the bin", ["red block"]),
    ("Grab two forks and a knife, then move the large blue mug onto the wooden shelf",
     ["fork", "fork #2", "knife", "large blue mug"]),
    ("Put both apples in the basket", ["apple", "apple #2"]),
])
def test_targets(task, expected):
    assert labels(task) == expected


def test_demo_task_fields():
    parsed = parse_task("Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork.")
    assert [t.id for t in parsed.targets] == [1, 2, 3, 4, 5]
    assert parsed.targets[0].noun == "cup" and parsed.targets[0].attributes == ("yellow",)
    assert parsed.targets[3].attributes == ()
    assert parsed.destinations == ["container"]
    assert parsed.nouns == ["cup", "spoon", "fork"]


def test_destination_with_attributes_is_excluded():
    # ep 60
    parsed = parse_task("Place the white bowl, blue bowls, yellow cups, pink cup, and blue cup "
                        "into the clear plastic container.")
    assert parsed.destinations == ["clear plastic container"]
    assert all(t.noun != "container" for t in parsed.targets)
    assert [t.label for t in parsed.targets] == [
        "white bowl", "blue bowl (all)", "yellow cup (all)", "pink cup", "blue cup"]


def test_open_ended_flag():
    parsed = parse_task("Place all cups and utensils into the container.")
    assert all(t.open_ended for t in parsed.targets)


def test_no_objects_raises():
    with pytest.raises(ValueError):
        parse_task("Do it now.")
