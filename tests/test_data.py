import pytest

from llm_jev.data import prompts
from llm_jev.data import sampling
from llm_jev.data import schema


def test_row_round_trip(example: schema.Example) -> None:
    assert schema.Example.from_row(example.to_row()) == example


def test_invalid_gold_is_rejected() -> None:
    with pytest.raises(ValueError, match="gold"):
        schema.Example("x", "s", (), "q?", ("a", "b"), gold=2)


def test_render_example_locates_parts(example: schema.Example) -> None:
    text, where = prompts.render_example(example, [2, 0, 1])
    ((start, end),) = where
    assert text[start:end] == example.parts[0].text
    assert (
        text.index("[1] Green") < text.index("[2] Blue") < text.index("[3] Red")
    )


def _rows(n_sources: int = 3, per: int = 10) -> list[schema.Example]:
    return [
        schema.Example(f"{s}:{i}", f"src{s}", (), "q?", ("a", "b"), 0)
        for s in range(n_sources)
        for i in range(per)
    ]


def test_dev_and_test_partition_the_split() -> None:
    rows = _rows()
    dev = sampling.select_part(rows, "dev", per=4, seed=0)
    test = sampling.select_part(rows, "test", per=4, seed=0)
    assert len(dev) == 12
    assert {r.id for r in dev} | {r.id for r in test} == {r.id for r in rows}
    assert not {r.id for r in dev} & {r.id for r in test}


def test_balanced_round_robin() -> None:
    rows = _rows(3, 10)
    picked = sampling.balanced(rows, 7, seed=0)
    counts = {
        s: sum(r.source == s for r in picked) for s in ("src0", "src1", "src2")
    }
    assert sorted(counts.values()) == [2, 2, 3]
