# Python Code Documentation Rules

**Core:** Comments explain *why*; docstrings define *contract*; type hints enforce *interface*.

---

## Comments

### Inline — only when code can't self-explain

```python
x = x.strip().lower()  # normalize for case-insensitive lookup         # ✅
x = x + 1              # increment x by one                            # ❌ restates code
name = clean(name)     # clean the name                                # ❌ restates code
pipe = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
(stdout, stderr) = pipe.communicate(timeout=30)
# Tool uses /dev/stderr for diag, stdout for JSON. Need both streams.  # ✅ explains non-obvious
result = parse_result(stdout, stderr)
```

### Block — design decisions, not mechanics

```python
# Legacy v1 API returns user_ids as strings; DB stores ints. Coerce.   # ✅ design rationale
user_ids = [int(uid) for uid in response.get("users", [])]

# Loop through users                                                   # ❌ restates code
for user in users: process(user)
```

### Markers — MUST have description, SHOULD have reference (issue/author/commit)

| Marker | Meaning |
|--------|---------|
| `TODO` | Planned work, not urgent |
| `FIXME` | Known defect, should fix |
| `NOTE` | Important context for reader |
| `HACK` | Temp workaround; pair w/ `TODO` |
| `XXX` | Danger — fragile/tricky |

```python
TODO(#284): deduplicate with build_index()              # ✅
FIXME(jdoe): off-by-one when count == 0                  # ✅
HACK: bypass SSL for staging (phased out Q3)             # ✅
# TODO: fix this later                                   # ❌ no ref
# FIXME: broken                                          # ❌ no ref
```

### Never

- Commented-out code — delete it; git history exists.
- Comments that restate code — if code can be self-documenting, do that first.
- Comments that explain confusing code instead of fixing it.

---

## Docstrings

Google-style per PEP 257. `"""triple double quotes"""`.

```
Summary line (imperative mood, ≤80 chars, ends with period).

Extended detail. Blank line after summary.

Args:
    param1: Description. Defaults to X.
    param2: Description.

Returns:
    Description. Type only when not obvious from annotation.

Raises:
    SomeError: When and why.

Examples:
    >>> result = func(param1="test")
    >>> assert result == expected
```

**Layout:** opening/closing `"""` on own lines (multi-line). Section headers: `Args:`, `Returns:`, `Raises:`, `Yields:`, `Examples:` — capitalized, colon, own line. Continuation 4‑space indent. Blank line between sections. Types in signature, NOT docstring.

### Per-symbol requirements

`✅` = required · `⬜` = optional · `❌` = omit. \
`*` = exempt for one-line docstrings (see below): Args and Returns/Yields may be omitted when the function is trivially obvious with no params worth documenting.

| Type | Args | Returns/Yields | Raises | Examples |
|------|------|---------------|--------|----------|
| Function w/ params | ✅\* | ✅ Returns\* | ⬜ if applicable | ⬜ |
| Function no params | ❌ omit | ✅ Returns | ⬜ | ⬜ |
| Generator w/ params | ✅* | ✅ Yields* | ⬜ if applicable | ⬜ |
| Generator no params | ❌ omit | ✅ Yields | ⬜ | ⬜ |
| Class | ❌ N/A | ❌ N/A | ❌ N/A | ⬜ |
| Module | ❌ N/A | ❌ N/A | ❌ N/A | ⬜ |
| `@property` | ❌ omit | ✅ | ⬜ | ❌ omit |
| Test function | ✅ | ⬜ | ⬜ | ⬜ |

**Generator return values:** If a generator also returns a value (`-> Generator[T, None, Ret]`), add a `Returns:` section documenting the return value. Rare; omit unless present in signature.

### One-line — only when trivially obvious (no params worth documenting, no ambiguity)

```python
def version() -> str: """Return the current library version string."""   # ✅
def is_empty(s: str | None) -> bool: """Return True if s is None/empty."""  # ✅
def parse_file(path, encoding="utf-8", errors="strict"): """Parse file."""  # ❌ conceals params
```

### Module — REQUIRED every `.py` file as first statement. Summarizes what module provides + usage.

### Class — REQUIRED every class. Attributes documented in docstring. One-liner OK for dataclass/NamedTuple/TypedDict.

### Function/method — REQUIRED public, SHOULD private (unless name+types self-explanatory).

### Overridden methods — omit docstring if contract identical. Use `@override` (3.12+) or `# override` comment (<3.12).

```python
from typing import override

class JSONHandler(BaseHandler):
    @override
    async def handle(self, request): ...  # inherits docstring from BaseHandler.handle  # ✅
```

### Docstring DON'Ts

- Signature repetition (`Args:` describes semantics, not names already in `def`)
- Implementation details (contract, not internals)
- Obvious side effects (`"opens a file"` in `open_file`)

---

## Type Annotations

Every public function: FULL annotations on ALL params + return type. Internal functions SHOULD.

| Target | Syntax |
|--------|--------|
| 3.12+ | PEP 695 (`def first[T]`), `type` alias stmt, `@override`, `[T: (int, float)]` constrained |
| 3.10+ | `X \| None`, `X \| Y` union syntax |
| <3.10 | `Optional[X]`, `Union[X,Y]`, `List[X]`, `TypeVar` fallback |

### Per-rule reference

| Rule | ✅ Good | ❌ Bad |
|------|---------|--------|
| Annotate all params + return | `def create(name: str, email: str) -> User:` | `def create(name, email, role="viewer"):` |
| `-> None` on void functions | `def notify(uid: str) -> None:` | `def notify(uid: str):` |
| `self`/`cls` unannotated; return `Self` typing (3.11+) | `def __enter__(self) -> Self:` (requires `from typing import Self`) | `def __enter__(self) -> "MyClass":` |
| Variable annotations for non-trivial types | `users: list[User] = cache.get(...)` | `count: int = 0` (trivial) |
| `list[X]` over `List[X]`, `dict[K,V]` over `Dict` (3.9+) | `def process(items: list[str]) -> dict[str, int]:` | `from typing import List, Dict` |
| `collections.abc` over `typing.Iterable` etc | `data: collections.abc.Iterable[str]` | `from typing import Iterable` |
| `X \| None` over `Optional[X]` (3.10+) | `def find(id: int \| None = None) -> User \| None:` | mixed idioms |
| `from __future__ import annotations` | when safe (NOT Pydantic/SQLAlchemy/FastAPI/runtime inspection) | breaks runtime annotation access |
| Protocol > ABC for structural subtyping | `class Drawable(Protocol): def draw(self, canvas): ...` | ABC when no impl sharing |
| TypedDict over `dict[str, Any]` | `class UserResponse(TypedDict): id: str` | `def get(u: str) -> dict[str, Any]:` |
| `Unpack[TypedDict]` for typed `**kwargs` (3.11+) | `def create(**kw: Unpack[UserParams])` | `def create(**kw: Any)` |
| `*args`/`**kwargs` element types | `*args: str`, `**kwargs: int \| str \| bool` | untyped variadic |
| Generics 3.12+ PEP 695 | `def first[T](items: list[T]) -> T \| None` | stale `TypeVar` unused |
| Generics <3.12 fallback | `_T = TypeVar("_T")`, `class Registry(Generic[_T])` | mixing PEP 695 + explicit TypeVar |
| Type aliases 3.12+ `type` stmt | `type JSON = dict[str, JSONValue]` | repeated inline unions |
| Type aliases <3.12 `TypeAlias` | `JSON: TypeAlias = dict[str, "JSONValue"]` | bare assignment |
| `Any` → avoid; use `object`, union, or `T` | `def load_pickle(path: str) -> object:` | `def transform(data: Any) -> Any:` |
| `# type: ignore` MUST include error code + justification | `# type: ignore[no-any-return]  # 3rd-party SDK lacks stubs, safe` | bare `# type: ignore` |
| Runtime checks only at untrusted boundaries | `isinstance(raw, dict)` on deserialized input | `isinstance(n, int)` inside `def double(n: int)` |

### Forward reference rules

- Use `from __future__ import annotations` UNLESS code uses Pydantic/FastAPI/SQLAlchemy/any runtime annotation inspection.
- When runtime inspection is needed: `TYPE_CHECKING` guard + string literal forward ref.

### `None` vs empty collections

Use `X | None` when `None` is semantically distinct from "empty". Default empty collection otherwise.

```python
def get_tags(pid: str) -> list[str] | None:  """Return tags or None if post DNE."""   # ✅
def get_tags(pid: str) -> list[str]:          """Return tags. [] means no tags."""      # ✅
```

---

## Anti-Patterns

| Anti-pattern | Why wrong | Fix |
|---|---|---|
| Narrative comments | Restate code; rot faster than code | Delete; make code self-documenting |
| Commented-out code | Clutter; git history exists | Delete |
| Bare markers (`# TODO: fix this`) | No context → permanent noise | Add description + reference |
| Bare `# type: ignore` | Suppresses all errors, including unintended | Add error code + justification |
| `Any` as default on all params/returns | Laziness masquerading as flexibility | Use specific type, `object`, or `T` |
| Stale type aliases | Defined once, never reused or drifted | Remove or keep alive |
| Signature docstrings (`func(a,b) -> list`) | PEP 257 discourages; introspection already shows sig | Document semantics |
| Args section repeats param names | `x (int): The x value.` — zero info | Describe meaning, not name |
| Missing `-> None` on void functions | Incomplete signature | Add `-> None` |
| `Optional` w/o semantic distinction | `str \| None = None` when `None` ≡ `""` | Use `str = ""` |
| Protocol methods w/ `raise NotImplementedError` | Wrong idiom | Use `...` literal |
| Docstring describes impl, not contract | "Uses dict to map..." — caller doesn't need internals | Describe behavior |
| Missing blank line after summary in multi-line docstring | Hard to parse | Always insert blank line |
| Unnecessary fallback syntax | `Optional[X]` / `List[X]` / `TypeVar` when targeting 3.12+ | Use modern form |

---

## Decision Checklist

Before writing a comment, docstring, or type annotation, ask:

1. **Comment explains *why*, not *what*?** Restates code → delete. Non-obvious decision → keep.
2. **Comment actionable?** TODO/FIXME has reference? No → add one or delete.
3. **Google-style docstring?** Summary → blank line → sections. No RST markup.
4. **Imperative mood?** "Connect" not "Connects" or "This function connects."
5. **All public symbols covered?** Function, method, class, module. Overrides w/ identical contract exempt.
6. **All params + return annotated?** `-> None` explicit on void functions.
7. **Modern syntax?** PEP 695, `X \| None`, `list[str]`, `type` alias. `from __future__ import annotations` when safe (not Pydantic/SQLAlchemy/FastAPI).
8. **`Any` justified?** Has comment explaining why? No → concrete type.
9. **Complex types aliased?** Repeated unions extracted? Check source + docstring.
10. **Annotation matches docstring?** Param names match signature? Returns agree? Mismatches = bugs.
