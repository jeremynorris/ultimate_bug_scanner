"""ubs_core.py_patterns.quality — categories 8, 9, 12, 14, 15, 22, 23 (bead 0xjg.5).

Faithful ports of the legacy rg pipelines in modules/ubs-python.sh:
- CATEGORY 8 FUNCTION & SCOPE (11351-11389): mutable default arguments,
  functions with >6 parameters, nested function declarations. The missing-
  returns heuristic (11383-11389) needs two project-wide counts and lives in
  ubs_core.py_detectors.missing_returns instead.
- CATEGORY 9 PARSING (11458-11463): string concatenation with `+` next to a
  quote. The json.loads heredoc is ubs_core.py_detectors.json_loads.
- CATEGORY 12 PERFORMANCE & MEMORY (11540-11562): string concat / re.compile /
  I/O inside loops.
- CATEGORY 14 CODE QUALITY MARKERS (11589-11615): TODO/FIXME/HACK/XXX.
  NOTE is measured for the legacy breakdown display only and never feeds the
  total, so it is not in the regex. The warning>20 / info>10 / info>0 ladder
  carries three legacy titles ("Significant technical debt" / "Moderate
  technical debt" / "Minimal technical debt"); like js.markers.todo-family,
  the titles merge into one neutral record-stream title.
- CATEGORY 15 REGEX & STRING SAFETY (11626-11637): nested quantifiers
  (ReDoS), dynamic re.compile from a bare variable.
- CATEGORY 22 PACKAGING (11887-11889): editable installs / local file:// refs.
  The requirements.txt unpinned-lines check (11881-11885) greps
  $PROJECT_DIR/requirements.txt directly — a file the v2 list never contains
  (.txt is outside INCLUDE_EXT) — and is a documented v2 divergence.
- CATEGORY 23 NOTEBOOK HYGIENE (11900-11902): embedded notebook outputs.

Equivalence notes:
- All sources are GREP_RN/GREP_RNI (ERE), never line-anchored unless the
  pattern itself had `^`; `[^)]`-style classes gain `\n` exclusions so a
  whole-text finditer cannot cross legacy line boundaries.
- Loop pipelines (`rg 'for...' | grep -A3 'X' | grep -w 'X'`) count exactly
  the lines containing BOTH the loop header and X, so they become
  MULTILINE-anchored composites; one line yields one match (count_lines
  counts lines). The `-w` filters keep their word-boundary semantics as
  lookarounds (`x+=1` without spaces never matched legacy either).
- Nested-defs: legacy stages 2+3 (`grep -A2 -E '^[[:space:]]+def[[:space:]]'`
  over path:line:text output) could never match — rg prefixes every line with
  `path:` — so the legacy check was dead and the info finding never fired.
  The port restores the evident intent (indented `def` lines).
- `^-e[[:space:]]+` in category 22 was equally dead behind the `path:` prefix;
  MULTILINE restores it for list-scoped requirement-like files.
- cat 9's counter was plain `wc -l` (no ubs:ignore strip) while the engine
  always skips marker lines — same accepted divergence class as the js port.
"""
from __future__ import annotations

import ast
import re

from ubs_core.py_scan import Pattern


def _repeated_io_lines(source: str) -> set[int] | None:
    """Refine the legacy same-line candidates without widening that policy.

    A loop word in prose, an identifier or a call's generator argument does
    not repeat the call. Multiline loop coverage remains a separate policy
    change: this filter only removes disproven legacy regex candidates.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        # Native Python syntax failures are reported by the completion gate;
        # retain conservative regex evidence for notebooks/Cython here.
        return None
    nodes = tuple(ast.walk(tree))
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    scope_kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)

    def scope_of(node):
        parent = parents.get(node)
        while parent is not None and not isinstance(parent, scope_kinds):
            parent = parents.get(parent)
        return parent

    bound = {node.id for node in nodes
             if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
    bound.update(node.arg for node in nodes if isinstance(node, ast.arg))
    bound.update(node.name for node in nodes
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    path_types = set()
    path_modules = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                if node.module == 'pathlib' and not node.level and alias.name == 'Path':
                    path_types.add(local)
                else:
                    bound.add(local)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split('.')[0]
                if alias.name == 'pathlib':
                    path_modules.add(local)
                else:
                    bound.add(local)
    path_types -= bound
    path_modules -= bound
    dynamic_bindings = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {'exec', 'eval', 'globals', 'locals', 'vars', 'setattr'}
        or isinstance(node, ast.ImportFrom) and any(alias.name == '*' for alias in node.names)
        or isinstance(node, ast.Name) and node.id == '__builtins__'
        or isinstance(node, ast.Attribute) and (
            node.attr in {'__dict__', '__globals__', 'f_globals', 'f_locals', 'f_builtins'}
            or node.attr in {'open', 'Path'} and isinstance(node.ctx, (ast.Store, ast.Del)))
        for node in nodes)

    def is_path_type(node):
        return (isinstance(node, ast.Name) and node.id in path_types
                or isinstance(node, ast.Attribute) and node.attr == 'Path'
                and isinstance(node.value, ast.Name) and node.value.id in path_modules)

    def typed_path(node, scope):
        if isinstance(node, ast.Call):
            return is_path_type(node.func)
        if not isinstance(node, ast.Name) or not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        arguments = {arg.arg: arg.annotation for arg in (
            *scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs)}
        writes = [item for item in nodes if isinstance(item, ast.Name)
                  and item.id == node.id and isinstance(item.ctx, (ast.Store, ast.Del))
                  and scope_of(item) is scope]
        if not writes:
            return is_path_type(arguments.get(node.id))
        # An element of a typed Path sequence is enough for the scanner's
        # normal `for path in files` read loop, but a later reassignment is not.
        if len(writes) != 1:
            return False
        loop = parents.get(writes[0])
        if not (isinstance(loop, (ast.For, ast.AsyncFor)) and loop.target is writes[0]
                and isinstance(loop.iter, ast.Name)):
            return False
        if any(isinstance(item, ast.Name) and item.id == loop.iter.id
               and isinstance(item.ctx, (ast.Store, ast.Del)) and scope_of(item) is scope
               for item in nodes):
            return False
        annotation = arguments.get(loop.iter.id)
        return (isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Name)
                and annotation.value.id in {'Sequence', 'Iterable', 'Iterator', 'list', 'tuple', 'set'}
                and is_path_type(annotation.slice))

    def read_only_opener(call, scope):
        if not isinstance(call, ast.Call) or dynamic_bindings:
            return False
        if isinstance(call.func, ast.Name) and call.func.id == 'open' and 'open' not in bound:
            mode_index = 1
        elif (isinstance(call.func, ast.Attribute) and call.func.attr == 'open'
              and typed_path(call.func.value, scope)):
            mode_index = 0
        else:
            return False
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            return False
        if any(keyword.arg in {None, 'opener'} for keyword in call.keywords):
            return False
        mode = call.args[mode_index] if len(call.args) > mode_index else ast.Constant(value='r')
        for keyword in call.keywords:
            if keyword.arg == 'mode':
                mode = keyword.value
        return isinstance(mode, ast.Constant) and mode.value in {'r', 'rb', 'rt'}

    def positive_size(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left, right = positive_size(node.left), positive_size(node.right)
            if left > 0 and right > 0:
                return min(left * right, 1 << 30)
        return 0

    def enclosing_loops(node):
        child = node
        parent = parents.get(child)
        while parent is not None and not isinstance(parent, scope_kinds):
            if isinstance(parent, (ast.For, ast.AsyncFor, ast.While)) and child in parent.body:
                yield parent
            child, parent = parent, parents.get(parent)

    def distinct_stream_each_iteration(loop, context, opener):
        # Chunking a file does not make reopening that same file for every
        # attempt cheap. Each enclosing loop must select the opened path;
        # a loop inside an already-open context cannot establish that fact.
        outer_contexts = set(enclosing_loops(context))
        if isinstance(opener.func, ast.Attribute):
            source = opener.func.value
        else:
            source = opener.args[0] if opener.args else next(
                (keyword.value for keyword in opener.keywords if keyword.arg == 'file'), None)
        if isinstance(source, ast.Call) and is_path_type(source.func) and len(source.args) == 1:
            source = source.args[0]
        while isinstance(source, ast.Attribute):
            source = source.value
        source_name = source.id if isinstance(source, ast.Name) else None
        outer_loops = tuple(enclosing_loops(loop))
        # A single path binding cannot establish independent progress through
        # two loops, including two nested loops that reuse the same name.
        if len(outer_loops) > 1:
            return False
        for outer in outer_loops:
            if not isinstance(outer, (ast.For, ast.AsyncFor)) or outer not in outer_contexts:
                return False
            targets = set(ast.walk(outer.target))
            selected = {target.id for target in targets
                        if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)}
            if source_name not in selected:
                return False
            for node in ast.walk(outer):
                if node in targets or not isinstance(getattr(node, 'ctx', None), (ast.Store, ast.Del)):
                    continue
                target = node
                while isinstance(target, (ast.Attribute, ast.Subscript)):
                    target = target.value
                if isinstance(target, ast.Name) and target.id == source_name:
                    return False
        return True

    def draining_read(loop):
        # Exempt a buffered, EOF-controlled drain of one stable opened reader.
        # A byte-at-a-time read, unknown size, cursor reset, alias escape or
        # reassignment is still a candidate; a literal size alone proves none
        # of those properties.
        test = loop.test
        if not (isinstance(test, ast.NamedExpr) and isinstance(test.target, ast.Name)
                and isinstance(test.value, ast.Call)):
            return None
        call = test.value
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'read'
                and isinstance(call.func.value, ast.Name) and len(call.args) == 1
                and not call.keywords and positive_size(call.args[0]) >= 8192):
            return None
        name = call.func.value.id
        context = parents.get(loop)
        while context is not None and not isinstance(context, scope_kinds):
            if isinstance(context, ast.With):
                item = next((item for item in context.items
                             if isinstance(item.optional_vars, ast.Name)
                             and item.optional_vars.id == name), None)
                if item is not None:
                    break
            context = parents.get(context)
        else:
            return None
        if not read_only_opener(item.context_expr, scope_of(context)):
            return None
        if not distinct_stream_each_iteration(loop, context, item.context_expr):
            return None
        loop_nodes = set(ast.walk(loop))
        for node in ast.walk(context):
            if not isinstance(node, ast.Name) or node.id != name or node is item.optional_vars:
                continue
            if not isinstance(node.ctx, ast.Load) or scope_of(node) is not scope_of(context):
                return None
            member = parents.get(node)
            invocation = parents.get(member)
            if not (isinstance(member, ast.Attribute) and member.value is node
                    and isinstance(invocation, ast.Call) and invocation.func is member):
                return None
            if node in loop_nodes and invocation is not call:
                return None
            if member.attr not in {'read', 'seek', 'tell', 'fileno', 'close'}:
                return None
        return call

    class RepeatedIO(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.callback_depth = 0
            self.stream_reads = set()
            self.lines = set()

        def visit_Call(self, node):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')
            root = func
            while isinstance(root, ast.Attribute):
                root = root.value
            if (self.depth and node not in self.stream_reads
                    and (name in {'open', 'read', 'write'}
                         or isinstance(root, ast.Name) and root.id == 'requests')):
                self.lines.add(node.lineno)
            self.generic_visit(node)

        def visit_For(self, node):
            # Iterables are evaluated once; callbacks passed to an unknown
            # iterable factory can still run repeatedly, so keep their reads.
            self.callback_depth += 1
            self.visit(node.iter)
            self.callback_depth -= 1
            self.depth += 1
            self.visit(node.target)
            for statement in node.body:
                self.visit(statement)
            self.depth -= 1
            for statement in node.orelse:
                self.visit(statement)

        visit_AsyncFor = visit_For

        def visit_While(self, node):
            read = draining_read(node)
            if read is not None:
                self.stream_reads.add(read)
            self.depth += 1
            self.visit(node.test)
            for statement in node.body:
                self.visit(statement)
            self.depth -= 1
            for statement in node.orelse:
                self.visit(statement)

        def visit_FunctionDef(self, node):
            for expression in (*node.decorator_list, *node.args.defaults, *node.args.kw_defaults):
                if expression is not None:
                    self.visit(expression)
            depth, self.depth = self.depth, 0
            callbacks, self.callback_depth = self.callback_depth, 0
            for statement in node.body:
                self.visit(statement)
            self.depth, self.callback_depth = depth, callbacks

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            depth = self.depth
            self.depth = max(self.depth, self.callback_depth)
            self.generic_visit(node)
            self.depth = depth

        def visit_ListComp(self, node):
            depth = self.depth
            for generator in node.generators:
                self.callback_depth += 1
                self.visit(generator.iter)
                self.callback_depth -= 1
                self.depth += 1
                self.visit(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
            self.depth = depth

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

    visitor = RepeatedIO()
    visitor.visit(tree)
    return visitor.lines

# One parameter: optional separator (`*,` / `/,`) first, then a name with
# its annotation and default. One level of brackets/parens is allowed so the
# commas inside `dict[str, int]` or `f(a, b)` stay inside the parameter.
_PARAM = (
    r"[ \t]*(?:[*/][ \t]*,[ \t]*)?(?![ \t]*[*/][ \t]*[,)])"
    r"(?:[^,()\[\]\n]|\[[^\[\]\n]*\]|\([^()\n]*\))+"
)

PATTERNS: list[Pattern] = [
    # ── Category 8: FUNCTION & SCOPE ISSUES ─────────────────────────────────
    Pattern(
        category=8,
        rule_id="py.functions.mutable-default",
        title="Mutable default arguments",
        regex=re.compile(
            r"def[ \t]+[A-Za-z_][A-Za-z0-9_]*\([^)\n]*=[ \t]*(?:\[\]|\{\}|set\(\))[^)\n]*\)"
        ),
        thresholds=((0, "critical"),),
    ),
    Pattern(
        category=8,
        rule_id="py.functions.high-param-count",
        title="Functions with >6 parameters",
        # Seven parameters on the `def` line, counted as parameters rather
        # than commas: a leading `self`/`cls` receiver is not one, nor is a
        # bare `*` or `/` separator, and a comma inside an annotation's
        # brackets (`dict[str, int]`, `tuple[Fact, ...]`) separates nothing.
        # Six parameters and a trailing comma at the end of the line mean the
        # list continues, so the seventh is on the next line.
        regex=re.compile(
            r"(?m)def[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*\([ \t]*"
            r"(?:(?:self|cls)[ \t]*(?::[^,()\n]*)?,|(?![ \t]*(?:self|cls)\b))"
            r"(?:" + _PARAM + r",){6}"
            r"(?:" + _PARAM + r"|[ \t]*(?:#[^\n]*)?$)"
        ),
        thresholds=((3, "warning"),),
    ),
    Pattern(
        category=8,
        rule_id="py.functions.nested-defs",
        title="Nested functions - verify closures and lifetimes",
        # Intent of the (dead) pipeline: indented `def name(` lines.
        regex=re.compile(r"^[ \t]+def[ \t]+[A-Za-z_][A-Za-z0-9_]*\(", re.MULTILINE),
        thresholds=((10, "info"),),
    ),
    # ── Category 9: PARSING & TYPE CONVERSION ───────────────────────────────
    Pattern(
        category=9,
        rule_id="py.parsing.string-concat-plus",
        title="String concatenation with +",
        regex=re.compile(r"\+[ \t]*['\"]|['\"][ \t]*\+"),
        thresholds=((5, "info"),),
        scan_strings=True,  # the quote character IS the evidence
        exclude_regex=re.compile(r"\+\+|[+\-]="),  # legacy `grep -v -E '\+\+|[+\-]='`
    ),
    # ── Category 12: PERFORMANCE & MEMORY ───────────────────────────────────
    Pattern(
        category=12,
        rule_id="py.perf.string-concat-loop",
        title="String concatenation in loops",
        # Line holds a `for|while ... :` header AND a word-bounded `+=`.
        regex=re.compile(
            r"^(?=.*(?:for|while)[ \t]+.*:)(?=.*(?<![A-Za-z0-9_])\+=(?![A-Za-z0-9_]))",
            re.MULTILINE,
        ),
        thresholds=((8, "info"),),
    ),
    Pattern(
        category=12,
        rule_id="py.perf.re-compile-loop",
        title="Regex compiled in loop",
        # Line holds a `for|while ... :` header AND a word-bounded re.compile.
        regex=re.compile(
            r"^(?=.*(?:for|while)[ \t]+.*:)(?=.*(?<![A-Za-z0-9_])re\.compile(?![A-Za-z0-9_]))",
            re.MULTILINE,
        ),
        thresholds=((0, "info"),),
    ),
    Pattern(
        category=12,
        rule_id="py.perf.io-in-loop",
        title="I/O in loops",
        # Retain the legacy same-line candidates and project-wide threshold;
        # the structural filter verifies that a real call repeats.
        regex=re.compile(
            r"^(?=.*(?:for|while))(?=.*(?:open\(|read\(|write\(|requests\.))",
            re.MULTILINE,
        ),
        thresholds=((5, "warning"),),
        code_only=True,
        line_filter=_repeated_io_lines,
    ),
    # ── Category 14: CODE QUALITY MARKERS ───────────────────────────────────
    Pattern(
        category=14,
        rule_id="py.code-quality.tech-debt",
        title="Technical debt markers",
        # NOTE: breakdown-only in legacy, never in the total. RNI = -i.
        regex=re.compile(r"TODO|FIXME|HACK|XXX", re.IGNORECASE),
        thresholds=((20, "warning"), (10, "info"), (0, "info")),
        case_insensitive=True,
    ),
    # ── Category 15: REGEX & STRING SAFETY ──────────────────────────────────
    Pattern(
        category=15,
        rule_id="py.regex.nested-quantifiers",
        title="Potential catastrophic regex",
        regex=re.compile(r"\([^)\n]*\+[^)\n]*\)\+|\([^)\n]*\*[^)\n]*\)\+"),
        thresholds=((3, "warning"),),
        scan_strings=True,  # a regex source is always a string literal
        # A rule table that quotes a catastrophic pattern as an *example* is
        # documenting it, not running it: `- pattern: String $R = "(a+)+";`
        # inside an ast-grep rule is a fixture, not a compiled regex.
        exclude_regex=re.compile(r"(?:^|[^A-Za-z0-9_])(?:pattern|regex):|- (?:pattern|regex)\b"),
    ),
    Pattern(
        category=15,
        rule_id="py.regex.dynamic-compile",
        title="Dynamic regex construction",
        regex=re.compile(r"re\.compile\([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*\)"),
        thresholds=((3, "info"),),
    ),
    # ── Category 22: PACKAGING & CONFIG HYGIENE ─────────────────────────────
    Pattern(
        category=22,
        rule_id="py.packaging.editable-local",
        title="Editable/local dependency references",
        # No legacy description (bare print_finding title only).
        regex=re.compile(r"^-e[ \t]+|file:/", re.MULTILINE),
        thresholds=((0, "info"),),
    ),
    # ── Category 23: NOTEBOOK HYGIENE ───────────────────────────────────────
    Pattern(
        category=23,
        rule_id="py.notebooks.outputs-embedded",
        title="Notebooks contain outputs",
        regex=re.compile(r'"outputs":[ \t]*\[[ \t]*\{'),
        thresholds=((5, "info"),),
        scan_strings=True,  # notebooks are JSON, not a Python token stream
    ),
]
