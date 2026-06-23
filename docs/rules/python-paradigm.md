# Python Programming Paradigm Rules

A convention for choosing and applying programming paradigms in Python 3.12+ codebases. Core principle: **Functions transform data; modules group functions; classes describe data. Avoid class inheritance.**

**Rule priority when sections conflict:** The Decision Checklist order reflects priority — data/behavior separation (items 1–3, 9, 21, 25) > syntax modernity (4) > purity/side-effects (5–6, 11, 13) > state discipline (7, 12) > patterns (8, 14, 15, 18) > module organization (10) > contracts (17, 19, 20) > tooling choices (decorators, logging, async — 16, 22–24). When two rules pull in opposite directions, the higher-priority rule wins.

## Python 3.12+ syntax enforcement

Use modern syntax. Do not fall back to pre-3.12 idioms unless a dependency pins the project to an older version.

**Generics (PEP 695):** `def first[T](items: list[T]) -> T | None` — not explicit `TypeVar`. PEP 695 supports `**P` for `ParamSpec` in function signatures. Exceptions for explicit `TypeVar`/`ParamSpec`: (1) a `ParamSpec` variable shared across multiple non-nested definitions (PEP 695 scopes type params to their declaration site); (2) generic utility classes that implement `__class_getitem__` for non-standard generic behavior.

```python
def first[T](items: list[T]) -> T | None: ...        # ✅ PEP 695
def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]: ...  # ✅ PEP 695 **P syntax

from typing import TypeVar; _T = TypeVar("_T")       # ❌ explicit when PEP 695 works
from typing import ParamSpec; P = ParamSpec("P")      # ✅ required only when sharing across multiple definitions
```

**Type aliases (PEP 695):** `type JSON = dict[str, JSONValue]` — not bare assignment.
**Override decorator (PEP 698):** `@override` on every nominal method override (explicit base-class inheritance). Structural Protocol matches (implicit duck-typing conformance) do not require `@override`.
**Enums:** `StrEnum`, `IntEnum` — not `class Status(str, Enum)`.
**Self type:** `-> Self` for methods returning `self`/`cls`.
**Stdlib baselines:** `list[str]` (not `List[str]`), `X | None` (not `Optional[X]`), `collections.abc` (not `typing` equivalents), `pathlib` for paths, `itertools.batched` for chunking, `tomllib` for TOML.
**`__future__` annotations:** Use unless the module uses Pydantic, FastAPI, or SQLAlchemy (which inspect annotations at runtime). This is orthogonal to PEP 695 generic syntax — use PEP 695 in all modules regardless of `__future__` annotations.

**Constrained vs bounded:** `[T: (int, float)]` means T must be exactly int or float (no subtypes). `[T: int | float]` means T can be int, float, bool, or any other subtype assignable to the union.

## Default to functions

Primary behavior unit is a **function**. Do not create a class solely to group related functions — use a module.

```python
# ✅ module-level function
def calculate_total(prices: list[float], tax_rate: float) -> float: ...

# ❌ class as namespace for a single function
class InvoiceCalculator:
    def calculate(self, prices: list[float], tax_rate: float) -> float: ...
```

**Exceptions:** Framework-mandated classes (Django views, Pydantic `BaseModel`, Starlette middleware) and **test classes** for pytest organization — these are framework idioms, not design choices. Keep the class thin; put all real logic in pure functions.

## Data modeling

Default: `@dataclass(frozen=True, slots=True)`. **Priority-ordered selection hierarchy:**

| Priority | Structure | Justification req'd |
|---|---|---|
| 1 (default) | `@dataclass(frozen=True, slots=True)` | No — always start here |
| 2 | `@dataclass` (mutable) | Yes — document why frozen won't work |
| 3 | `NamedTuple` | Yes — tuple unpacking boundary |
| 4 | `TypedDict` | Yes — only at I/O boundaries |
| 5 | `StrEnum`/`IntEnum` | No — correct for closed sets |
| 6 | `NewType` | No — correct for its niche |

**Local-data exemption:** Data types defined with a `_`-prefixed name that never leave their defining module may use mutable `@dataclass` without justification. If the type is imported by another module, freeze it.

```python
@dataclass(frozen=True, slots=True)  # ✅ default
class Point:
    x: float
    y: float

@dataclass  # ✅ local-data exemption (never leaves this module)
class _InternalBuffer:
    items: list[str]
    cursor: int
```

**No methods beyond `__post_init__`** (invariant enforcement) and **dunder methods** (operator/protocol integration). `__post_init__` is invariant enforcement at construction time — must be pure, no IO, no external mutation.

**Dunder methods** are the one exception to "no behavior on data." Constraints: (1) domain-meaningful, (2) pure (no IO, no mutation of `self`), (3) consistent (pair `__eq__` with `__hash__`, use `order=True` for whole-ordering).

```python
@dataclass(frozen=True, slots=True)
class Vector:
    x: float
    y: float
    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)
```

## No inheritance

**Concrete inheritance is discouraged** — prefer composition, `Protocol`, higher-order functions, and pattern matching.

| Instead of inheritance | Use |
|---|---|
| Shared interface | `typing.Protocol` for structural subtyping |
| Shared implementation | Composition — call another function/module |
| Type-based dispatch | `match/case` pattern matching |
| Extension via hooks | Accept callback/strategy functions as parameters |

```python
class Drawable(Protocol):
    def draw(self, canvas: Canvas) -> None: ...

class Dog(Animal): ...  # ❌ concrete inheritance
```

**Shallow inheritance** (≤2 levels, parent has no mutable state, child only extends behavior) is acceptable only when the parent is from a third-party library that requires inheritance for its plugin/extension mechanism — document why composition doesn't fit.

**Allowed exceptions:** Framework-imposed base classes (Pydantic `BaseModel`, `Exception`, Django `Model`, Starlette middleware). Test: if removing the parent would require rewriting the framework integration, it's framework-imposed. Keep business logic in pure functions regardless.

## Immutability

Prefer immutable data. Mutation only when localized, scoped, and justified.

```python
def promote(task: Task) -> Task:
    return dataclasses.replace(task, priority=task.priority + 1)  # ✅

def append_error(errors: list[str], msg: str) -> None:
    errors.append(msg)  # ❌ mutates caller's list
```

**Acceptable mutation:** (1) local accumulators, (2) perf-critical hot paths (document why), (3) algorithmic necessity (isolate in pure wrapper), (4) caching via `functools.cache`.

**Caveat — frozen dataclasses with collection fields:** `frozen=True` prevents field reassignment but does **not** prevent mutation of the field's value (e.g., `self.items.append(x)` is silently permitted). This is not "acceptable mutation" — it's an aliasing risk. Mitigate by using `tuple` for truly immutable collections, returning copies from accessors, or documenting the aliasing behavior. Do not use this as a justification to skip `frozen=True`.

## Pure functions and side-effect isolation

Core logic is pure. Side effects at module/application edges.

```python
def calculate_discount(price: float, tier: str) -> float: ...  # ✅ pure

def calculate_and_save(price, tier, db):
    db.log(price)  # ❌ hidden side effect in "calculate" function
```

**Pattern:** `IO edge → pure logic → IO edge`.
**Naming:** Impure functions should name the IO — `read_`, `write_`, `save_`, `load_`. Pure functions should not imply IO.

## Async and concurrency

Async functions are inherently impure (resumable state machines, scheduling-dependent). Keep the pure core sync; async adapters at the edge.

```python
def process_order(order: Order) -> Receipt: ...              # ✅ pure sync
async def handle_order(req, db):
    receipt = process_order(Order(req.json()))                # pure
    await db.save(receipt)                                    # edge IO
```

**Rules:** (1) No `async def` that uses none of `await`, `async for`, `async with`, or `yield` in an async generator — if none of these appear, make it sync. (2) CPU-bound work off the event loop via `asyncio.to_thread`. (3) Prefer `asyncio.TaskGroup` over `asyncio.gather` for structured concurrency. (4) Keep async surface minimal — async infection (converting everything because one function awaits) is an anti-pattern.

## Logging and observability

Logging is a side effect. Structure it so the pure core stays testable.

**Preferred: edge logging** — log after pure functions, not inside them.
**Alternative: logger as dependency** — `logger: Logger | None = None`, resolve lazily.
**Alternative: observation return** — `-> tuple[Result, list[Observation]]` for audit trails.
**Cross-cutting:** `@logged` / `@timed` decorators.

```python
def process(data: RawData) -> Processed: ...            # ✅ pure — no logging
def handle(raw: RawData) -> Response:
    result = process(raw)
    logger.info("processed %d records", result.count)   # edge logging
    return Response(result)
```

Use structured logging (`structlog` or JSON formatter). String-format logs are not queryable.

## Custom decorators

Decorators are higher-order functions. They must be transparent, composable, and limited to cross-cutting concerns.

**Rules:** (1) Preserve identity — always `@wraps`. (2) No decoration-time side effects (no IO at `@decorator` time). (3) Compose — document ordering constraints if any. (4) Verb-based names (`@logged`, `@timed`). (5) Parameterized decorators are factories — resolve config per-call, not at decoration. (6) No silent exception swallowing — re-raise or document new contract in name (`@fallback_to(default)`).

```python
def logged[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        logger.debug("calling %s", func.__name__)
        return func(*args, **kwargs)
    return wrapper
```

## State management

State is explicit: threaded through parameters and returned as values. No `global`, no mutable module-level state.

```python
def run_batch(items: list[Item], initial: State) -> list[Result]:
    state = initial
    results: list[Result] = []
    for item in items:
        result, state = process_one(item, state)
        results.append(result)
    return results

current_state: State = State()           # ❌ global mutable state
```

**Acceptable patterns:** Local function state, generator state across `yield`, context manager scoped state, `functools.cache`, dependency-injected state stores.

## Pattern matching

Prefer `match/case` over `isinstance` chains, visitor patterns, and manual destructuring.

```python
def handle(event: Event) -> str:
    match event:
        case UserCreated(id=uid): return f"created {uid}"
        case UserDeleted(id=uid): return f"deleted {uid}"
        case _: return f"unknown: {event}"

# ❌ isinstance chain
if isinstance(event, UserCreated): ...
elif isinstance(event, UserDeleted): ...
```

**Use guards, enum literal patterns, and destructuring.** For exhaustiveness, use `assert_never()` (type-checked + runtime assertion) in the wildcard branch. Use `TYPE_CHECKING` + `assert_never()` when no runtime guard is desired — note this removes the fail-fast safety net for refactoring mistakes.

## Design by Contract

Every function has a contract: preconditions, postconditions, invariants. Make them explicit; fail fast on violation.

**Preconditions:** What must be true before the function runs. Fail fast — raise, don't return `None` or silently correct.

```python
def divide(n: float, d: float) -> float:
    if d == 0.0:
        raise ValueError("denominator must be non-zero")
    return n / d
```

Checking patterns (priority): type annotation → guard clause → `assert` (debug only) → docstring `Pre:` section.

**Postconditions:** What the function guarantees. Document in docstring `Post:` section. Assert critical ones in debug mode.

```python
def sort_descending(items: list[float]) -> list[float]:
    result = sorted(items, reverse=True)
    assert len(result) == len(items)  # postcondition: length preserved
    return result
```

**Invariants:** Conditions that always hold. Enforce in `__post_init__` for dataclasses.

```python
@dataclass(frozen=True, slots=True)
class Interval:
    start: float
    end: float
    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("start must be <= end")
```

**Negative-space assertions:** Document what cannot happen. `assert_never(x)` for exhaustiveness, `assert False, "unreachable"` for guarded wildcards, `assert x is not None` after a narrowing guarantee.

| Assertion | Enabled | Use for |
|---|---|---|
| `assert cond` | Debug only (`-O` disables) | Invariants, unreachable branches |
| `if not cond: raise V(...)` | Always | Public API preconditions |
| `assert_never(x)` | Static + runtime (`AssertionError`) | Match exhaustiveness — use in bare wildcard |
| `if TYPE_CHECKING: assert_never(x)` | Type-checked only (zero runtime cost) | Match exhaustiveness when fail-fast not needed |

## Error handling strategy

| Failure type | Mechanism | Example |
|---|---|---|
| Precondition violation | `raise` (fail fast) | `ValueError` |
| Expected absence | `T \| None` | `find_user → User \| None` |
| Domain failure | Custom `Exception` subclass | `class PaymentDeclined(Exception):` |
| Exhaustive handling | `match/case` on exception | `except PaymentDeclined as e:` |
| Cross-boundary errors | `Result[T, E]` | System boundaries only |

**Rules:** (1) Precondition violations raise — no silent correction, no `None` return. (2) Expected absence returns `T | None` — not an exception. (3) Domain failures are custom exceptions — callers distinguish without string parsing. (4) Try/dispatch with `match/case` — structured, exhaustive. (5) Result types at system boundaries only — not for intra-process logic. (6) Custom exceptions inherit from `Exception`, not `BaseException`; build a hierarchy for related failure modes.

## Module organization

One module = one responsibility. Modules export functions, data types, and constants.

**Structure:** docstring → imports (stdlib / third-party / local) → constants → data types → functions.
**Size:** <500 lines typical. Data-type modules split by domain concept, not line count.
**Exports:** `__all__` in library modules; `_`-prefix convention in application code.

```python
# invoice.py
"""Invoice domain: creation, validation, and formatting."""

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Invoice: ...

def create(items: list[Item]) -> Invoice: ...
def validate(inv: Invoice) -> list[str]: ...
def format(inv: Invoice) -> str: ...
```

## Testing implications

These are consequences of the paradigm, not new requirements.

**Pure functions → no mocking.** Test with real values. Every pure function with branching logic or non-trivial computation gets a unit test.

```python
def test_calculate_discount() -> None:
    assert calculate_discount(100.0, "gold") == 80.0
```

**Dataclass data → inline construction.** No fixture classes for data. Fixtures only for IO resources (DB connections, temp dirs).

**Property-based testing for invariants.** `@given` strategies for pre/post conditions — the natural complement to Design by Contract.

**Edge IO tests are thin.** One test per IO function with real infrastructure (temp files, test DBs). All logic behind the boundary is tested via pure function unit tests.

**Organization:** One `test_<module>.py` per `<module>.py`. Pure tests first (no fixtures), edge tests below (with fixtures).

## Anti-Patterns

- **Service classes** — `CustomerService` with a single method. Use module-level functions.
- **Static method namespaces** — `class Utils:` with `@staticmethod` definitions. Use a module.
- **Data class with behavior** — `validate()`, `to_json()`, `save()` on a `@dataclass`. Data is shape; functions operate on it.
- **Concrete inheritance** — `class Dog(Animal):`. Use composition, Protocol, pattern matching.
- **Hidden side effects** — `calculate_` that writes to disk. Name the IO or split pure/impure.
- **Mutable default arguments** — `def f(items=[])`. Use `None` + guard.
- **Unnecessary pre-3.12 generics** — explicit `TypeVar` when `def func[T]()` works.
- **Bare TypeAlias** — `JSON: TypeAlias = dict[...]` instead of `type JSON = dict[...]`.
- **NamedTuple over dataclass** — unless tuple unpacking at a boundary is needed.
- **TypedDict for internal data** — only at IO boundaries. Use dataclass internally.
- **Impure dunders** — `__str__` that writes to a log file. Dunders must be pure.
- **Inconsistent `__eq__`/`__hash__`** — custom `__eq__` without `__hash__` = unhashable.
- **`__iadd__` on frozen dataclasses** — implies mutation. Use `__add__` returning new instance.
- **Global mutable state** — module-level variables mutated via `global`. Thread explicitly.
- **Decorator signature loss** — no `@wraps`. Preserve identity.
- **Decoration-time side effects** — IO at import time. Side effects belong in the wrapper.
- **Exception-swallowing decorators** — `@safe` that catches everything and returns `None`.
- **Unchecked preconditions** — silent wrong results instead of fail-fast raise.
- **Untrusted input via `assert`** — `assert` is disabled under `-O`. Use `if/raise` for validation.
- **Missing exhaustiveness** — bare `case _:` on an enum match. Use `assert_never()`.
- **Invariant drift** — non-frozen dataclass whose invariant is bypassed by direct attribute assignment. Use `frozen=True`.
- **Bare `except:`** — catches `KeyboardInterrupt`. Use `except Exception:` at minimum.
- **Logging inside pure functions** — `calculate_` that logs. Edge logging only.
- **String-format logging** — `f"processed {n}"`. Use structured logging.
- **Exception as control flow** — raising and catching in the same function where `if` suffices.
- **Async on pure functions** — `async def` with no `await`. Make it sync.
- **Blocking the event loop** — CPU-bound work in `async def` without `to_thread`.
- **Async infection** — converting everything because one function awaits.
- **Over-mocking pure functions** — `MagicMock()` on a pure function. Use real values.
- **Flat exception hierarchy** — `raise Exception("msg")` without subclasses. Callers need to distinguish.
- **Result types for intra-process logic** — `Ok`/`Err` inside one function. Reserve for boundaries.

## Decision Checklist

Before adding a new type, function, or module, ask:

1. **Is this behavior?** → Write a function. (Not a method, not a class.)
2. **Is this data?** → Start with `@dataclass(frozen=True, slots=True)`. Justify alternatives.
3. **Does this seem like inheritance?** → Discouraged. Use Protocol, composition, HOFs, pattern matching.
4. **Does this function use pre-3.12 syntax?** → Enforce PEP 695 generics, `type` alias, `@override`.
5. **Does this function have side effects?** → Push to caller or name to reflect IO. Keep pure core separate.
6. **Does this function mutate an argument?** → Return new value instead. Document if unavoidable.
7. **Does this use global/module-level mutable state?** → Thread explicitly through parameters.
8. **Is this an `isinstance` chain?** → Replace with `match/case`.
9. **Is this a class with no instance state?** → Replace with module-level functions.
10. **Is this module doing more than one thing?** → Split (<500 lines typical).
11. **Is this `async def` without `await`, `async for`, `async with`, or async generator `yield`?** → Make it sync. Keep async surface minimal.
12. **Is any `self` attribute mutated outside `__init__`?** → Return new instance instead. (Exception: `__post_init__` invariant enforcement.)
13. **Does a function name imply purity when it has side effects?** → Rename or refactor.
14. **Is this a concrete inheritance relationship?** → Can Protocol or composition replace it?
15. **Is this a decorator?** → `@wraps`, no decoration-time side effects, verb name, documented ordering.
16. **Is logging happening in a pure function?** → Move to edge. Pure function behavior must not change when logging is removed.
17. **Is a precondition missing?** → Guard clause or type annotation. Fail fast.
18. **Is this a `match/case` wildcard?** → Could it silently ignore a new variant? Use `assert_never()`.
19. **Is `assert` used for validation?** → Use `if/raise` for production checks. `assert` is for invariants only.
20. **Is this a precondition violation, expected absence, or domain failure?** → Raise, `T | None`, custom exception respectively.
21. **Is this a `NamedTuple`/`TypedDict`?** → Why not a dataclass? Justify JSON-boundary or tuple-unpacking need.
22. **Is this test mocking a pure function?** → Use real values. No mocks needed for pure functions.
23. **Is this a `Result` type for intra-process logic?** → Prefer exceptions. Result at boundaries only.
24. **Is this an `asyncio.gather` call?** → Prefer `TaskGroup` for structured concurrency.
25. **Does this dataclass have a dunder?** → Pure, consistent, domain-meaningful. No IO, no `self` mutation.