# Naming Conventions

**Core:** Name the thing the way the reader would say it.

---

## Functions — Use `verb_noun` imperative order

```python
create_user(email, role)     # ✅
parse_file(filepath)         # ✅
user_create(...)             # ❌ domain-first
email_send(...)              # ❌ domain-first
```

Verb is the reusable action vocabulary; noun is the target. A verb earns a vocabulary slot by plausible recurrence, not current count: `authenticate_user` is OK (if you can name 2+ other nouns it would pair with, e.g., `authenticate_session`).

**Async:** Same `verb_noun`. No `_async` suffix unless sync variant is also public. Prefer `fetch_` for async I/O targeting **remote** sources specifically — `read_`/`write_` for local async I/O.

## Predicates — Use `is_`/`has_`/`can_`/`should_`/`needs_`/`will_`/`contains_`

| Prefix | Question |
|--------|----------|
| `is_` | State or identity |
| `has_` | Possession or containment |
| `can_` | Capability or permission |
| `should_` | Recommendation or policy |
| `needs_` | Requirement |
| `will_` | Future state |
| `contains_` | Collection membership |

```python
is_authenticated()    # ✅
has_permission(u, a)  # ✅
can_write(path)       # ✅
privileged            # ❌ bare adjective
running               # ❌ bare adjective
```

**Boolean `@property` tiebreaker:** When a `@property` returns `bool`, predicate naming wins over noun-attribute rule. Use `is_`/`has_`/`can_` prefix even on properties: `config.is_debug` ✅, not `config.debug` or `config.is_debug()`. The predicate clarity rule beats the noun rule when type is `bool`.

## Values and attributes — Use nouns (things the caller has, not actions)

```python
task.status          # ✅
session.metadata     # ✅
task.status()        # ❌ verb_noun on a value
```

Computed data w/o side effects → `@property` with noun name. Expensive or side-effectful → method with `verb_noun`.

## Retrieval verbs — Encode source + trust level

| Verb | Source | Trust meaning |
|------|--------|---------------|
| `get_` | Local state, cache, DB | Recorded fact / current local state |
| `fetch_` | Remote target (network, API, device) | Live data from target |
| `find_` | Collection w/ search/filter | Candidate(s); may be `None` |
| `query_` | Structured DB or index | Result set matching criteria |
| `list_` | Persisted collection | All available; no filtering |
| `match_` | Reference/lookup table | Deterministic matched result |

```python
get_user(id)        # local key lookup     # ✅
fetch_page(url)     # remote fetch         # ✅
find_customer(n,e)  # search criteria      # ✅
match_record(p,c)   # deterministic lookup # ✅
get_page(url)       # remote via local verb# ❌ verb trust mismatch
```

Do not blur. Once a verb's semantics are chosen, don't reuse it for a different source/trust level. Add project-specific verbs (`resolve_`, `lookup_`) as needed — key is internal consistency.

## Pluralization

- Single known entity → singular: `get_user(id)`
- Collection / multiple candidates → plural: `get_users()`, `find_matches(p)`
- `list_` signals enumeration (no filtering): `list_users()` vs `get_users()` (retrieve known set)

## Local and internal names

Internal code is not exempt. Use descriptive names beyond tiny scopes:

```python
session_manager     # ✅  sm           # ❌
normalized_code     # ✅  norm         # ❌
output_max_chars    # ✅  omax         # ❌
```

Short names OK when idiomatic and short-lived: `i`, `db`, `url`.

Private helpers get `_` prefix but still follow the language rule:

```python
_resolve_queries()   # ✅
```

The `_` signals "implementation detail" — not an exemption from descriptiveness.

Use `__` prefix (double underscore, name-mangled) only in class hierarchies where subclass shadowing is a concern — e.g., a method that parent children must not override. Otherwise, prefer `_`.

## Classes, constants, modules, exceptions — PEP 8

| Entity | Convention | Example |
|--------|-----------|---------|
| Classes | `CamelCase` | `UserSession`, `ConfigParser` |
| Constants | `UPPER_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Modules/packages | short lowercase, underscores when readability demands | `auth`, `utils`, `rate_limiter` |
| Exceptions | `CamelCase` + `Error`/`Warning` | `ValidationError`, `TimeoutWarning` |
| Type variables | short `CamelCase` | `T`, `TKey`, `TResponse` |

## Module surfaces — Call site reads naturally

```python
find_user(db, uid)        # ✅ verb_noun
format_report(recs,style) # ✅ verb_noun
sessions                  # ❌ noun, not action
spawn                     # ❌ creation verb needs a target; prefer `spawn_worker`
```

Exceptions to `verb_noun` OK when bare verb is a well-known domain operation and a specific noun would mislead (verb applies equally to multiple types). Criterion: bare verbs are acceptable for **actor-initiated operations on managed resources**; avoid bare verbs for **creation** (`spawn`, `create`, `build`) which need a target noun:

```python
kill        # ✅ kills jobs, futures, sessions — actor-initiated op
run         # ✅ runs tasks, pipelines, jobs
stop        # ✅ stops services, processes, timers
kill_job    # ❌ over-specific — only kills jobs?
spawn       # ❌ creation verb; callers need `spawn_worker(n=3)`
```

## Public API contract

Public/shared interfaces MUST have:
- Full type hints on every parameter
- Return type annotation
- Docstring first line: when to use it
- One-line `Returns:` section describing return shape

Internal helpers and private functions may omit at team discretion.

---

## Anti-Patterns

| Anti-pattern | Why wrong | Fix |
|---|---|---|
| Domain-first names (`user_create`) | Reverses imperative order | `verb_noun` |
| Bare predicates (`privileged`, `running`) | Ambiguous w/o prefix | Add `is_`/`has_`/`can_` |
| Shadowing builtins (`list`, `format`) | Overwrites builtin name | Rename |
| Context-encoding (`admin_reset_tokens`, `v2_process_order`) | Encodes deployment context, not function | Name what it does |
| Single-noun verbs (`grab_record`, `heal_connection`) | Verb plausible for only one noun | Use standard verb (`fetch_`, `fix_`) |
| Ambiguous abbreviations (`sm`, `obs`, `resp`) | Forces search to decode | Spell it out |

---

## Decision Checklist

Before adding or renaming a symbol, ask:

1. **Call site reads like a sentence?** If not, rewrite.
2. **Function or command?** Use `verb_noun`.
3. **Predicate?** Use `is_`/`has_`/`can_`/`should_`/`needs_`/`will_`/`contains_`.
4. **Value or attribute?** Use a noun.
5. **Retrieval?** Choose `get_`/`fetch_`/`find_`/`query_`/`list_`/`match_` by source + trust.
6. **Collection return?** Plural noun or `list_` prefix.
7. **Verb reusable?** Plausible across future nouns?
8. **Clear w/o comments?** Would the name survive stripped context?
9. **Consistent with project patterns?** If not, strong reason to deviate?
10. **Public API?** Add type hints, return type, docstring contract.
