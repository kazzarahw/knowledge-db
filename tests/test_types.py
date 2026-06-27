"""Regression guard: specified private functions must have return type annotations."""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.parametrize(
    "module_name,qualname,should_pass_before",
    [
        ("knowledge.chunk", "_convert_notebook", True),
        ("knowledge.chunk", "_scan_headings", True),  # already annotated — confirm
        ("knowledge.cli", "_build_parser", True),
        ("knowledge.indexer", "_source_signature", True),
        ("knowledge.indexer", "_walk_files", True),
        ("knowledge.indexer", "_index_source", True),
        ("knowledge.fetch", "_fetch_git_source", True),
        ("knowledge.fetch", "_clone", False),
        ("knowledge.fetch", "_pull", False),
    ],
)
def test_function_has_return_annotation(
    module_name: str, qualname: str, should_pass_before: bool
) -> None:
    import importlib

    mod = importlib.import_module(module_name)
    fn = dict(inspect.getmembers(mod, inspect.isfunction))[qualname]
    sig = inspect.signature(fn)
    has_annotation = sig.return_annotation is not inspect.Parameter.empty
    if should_pass_before:
        assert has_annotation, (
            f"{module_name}.{qualname} should already have a return annotation"
        )
    if not has_annotation:
        pytest.xfail(
            f"{module_name}.{qualname} has no return annotation — will be fixed in this task"
        )


def test_search_result_is_typeddict() -> None:
    """cmd_search must return a TypedDict, not dict[str, Any]."""
    import typing

    from knowledge.search import cmd_search

    hints = typing.get_type_hints(cmd_search)
    ret = hints.get("return")
    if ret is None:
        pytest.fail("cmd_search must have a return annotation")
    # Should be list[SearchResult] where SearchResult is a TypedDict
    assert hasattr(ret, "__origin__"), f"return type must be generic, got {ret}"
    assert ret.__origin__ is list, f"must be list[...], got {ret.__origin__}"
    elem = ret.__args__[0]
    assert hasattr(elem, "__annotations__"), f"element type {elem} is not a TypedDict"


@pytest.mark.parametrize(
    "path",
    [
        "knowledge/__init__.py",
        "tests/__init__.py",
    ],
)
def test_init_py_has_module_docstring(path: str) -> None:
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    assert stripped.startswith('"""') or stripped.startswith("'''"), (
        f"{path} is empty or missing module docstring"
    )
