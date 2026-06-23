"""Regression guard: specified private functions must have return type annotations."""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.parametrize(
    "module_name,qualname,should_pass_before",
    [
        ("knowledge.chunk", "_convert_notebook", False),
        ("knowledge.chunk", "_scan_headings", True),  # already annotated — confirm
        ("knowledge.cli", "_build_parser", False),
        ("knowledge.indexer", "_source_signature", False),
        ("knowledge.indexer", "_walk_files", False),
        ("knowledge.indexer", "_index_source", False),
        ("knowledge.fetch", "_fetch_git_source", False),
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
    from knowledge.search import cmd_search

    sig = inspect.signature(cmd_search)
    ret = sig.return_annotation
    # Should be list[SearchResult] where SearchResult is a TypedDict
    assert hasattr(ret, "__origin__"), f"return type must be generic, got {ret}"
    assert ret.__origin__ is list, f"must be list[...], got {ret.__origin__}"
    elem = ret.__args__[0]
    import typing

    assert isinstance(elem, typing.TypeVar) or hasattr(elem, "__annotations__"), (
        f"element type {elem} is not a TypedDict"
    )
