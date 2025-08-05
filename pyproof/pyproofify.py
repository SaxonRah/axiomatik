#!/usr/bin/env python3
"""
pyproofify.py – generate a PyProof-instrumented copy of a project
Usage:
    python pyproofify.py src/ proofed/ --loops --ghost --asserts --temporal --protocols --contracts

Testing (ran from 'py-proof root folder'):
    python pyproof/pyproofify.py pyproof/test/  proofed/  --all
"""
import shutil, click
from pathlib import Path
import libcst as cst
from libcst import metadata
from libcst.metadata import ScopeProvider, FunctionScope
from typing import List, Dict, Set, Optional, Union


class ContractInjector(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self, *,
                 loops: bool = False,
                 ghost: bool = False,
                 asserts: bool = False,
                 temporal: bool = False,
                 protocols: bool = False,
                 contracts: bool = False):
        super().__init__()
        self.loops = loops
        self.ghost = ghost
        self.asserts = asserts
        self.temporal = temporal
        self.protocols = protocols
        self.contracts = contracts

        # State tracking
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.class_methods: Dict[str, List[str]] = {}
        self.detected_protocols: Set[str] = set()
        self.sensitive_patterns = {'password', 'secret', 'key', 'token', 'credential'}

    # ---------------------------------------------------------
    # 1. Functions / methods - contract generation
    # ---------------------------------------------------------
    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        """Track class context for protocol detection"""
        self.current_class = node.name.value
        self.class_methods[self.current_class] = []
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef, updated: cst.ClassDef) -> cst.ClassDef:
        """Exit class context and add protocol decorators if detected"""
        class_name = self.current_class
        methods = self.class_methods.get(class_name, [])

        # Detect common protocols
        if self.protocols:
            if self._is_file_like_protocol(methods):
                self.detected_protocols.add(f"{class_name}_file_protocol")
            elif self._is_state_machine_protocol(methods):
                self.detected_protocols.add(f"{class_name}_state_protocol")

        self.current_class = None
        return updated

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        """Track function context"""
        self.current_function = node.name.value
        if self.current_class:
            self.class_methods[self.current_class].append(node.name.value)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated: cst.FunctionDef) -> cst.FunctionDef:
        """Add comprehensive function instrumentation"""
        new_decorators = list(updated.decorators)

        # 1. Add auto_contract decorator
        if self.contracts:
            auto_contract_decorator = cst.Decorator(
                decorator=cst.Attribute(
                    value=cst.Name("pyproof"), attr=cst.Name("auto_contract")
                )
            )
            new_decorators.append(auto_contract_decorator)

        # 2. Add protocol decorators for detected patterns
        if self.protocols and self.current_class:
            protocol_decorator = self._create_protocol_decorator(updated.name.value)
            if protocol_decorator:
                new_decorators.append(protocol_decorator)

        # 3. Instrument function body
        new_body = self._instrument_function_body(updated.body, updated.name.value)

        self.current_function = None
        return updated.with_changes(decorators=new_decorators, body=new_body)

    def _instrument_function_body(self, body: cst.BaseSuite, func_name: str) -> cst.BaseSuite:
        """Add comprehensive instrumentation to function body"""
        if isinstance(body, cst.IndentedBlock):
            statements = list(body.body)
            new_statements = []

            # Add ghost state initialization at function start
            if self.ghost:
                ghost_init = self._create_ghost_init_statement(func_name)
                new_statements.append(ghost_init)

            # Add temporal event recording at function start
            if self.temporal:
                temporal_start = self._create_temporal_event_statement(f"{func_name}_start")
                new_statements.append(temporal_start)

            # Process existing statements
            for stmt in statements:
                new_statements.append(stmt)

            # Add temporal event recording at function end (before return)
            if self.temporal:
                temporal_end = self._create_temporal_event_statement(f"{func_name}_end")
                # Insert before last statement if it's a return
                if (new_statements and
                        isinstance(new_statements[-1], cst.SimpleStatementLine) and
                        any(isinstance(s, cst.Return) for s in new_statements[-1].body)):
                    new_statements.insert(-1, temporal_end)
                else:
                    new_statements.append(temporal_end)

            return cst.IndentedBlock(body=new_statements)

        return body

    # ---------------------------------------------------------
    # 2. Assert to require conversion - FIXED for LibCST compatibility
    # ---------------------------------------------------------
    def leave_Assert(self, original_node: cst.Assert, updated: cst.Assert) -> cst.BaseSmallStatement:
        """Convert assert statements to require() calls"""
        if not self.asserts:
            return updated

        # Extract the test condition
        test_expr = updated.test

        # Create the claim string - properly escaped
        claim = self._expr_to_string(test_expr)
        # Escape quotes in the claim
        escaped_claim = claim.replace('"', '\\"').replace("'", "\\'")

        # Build require() call
        require_call = cst.Call(
            func=cst.Attribute(value=cst.Name("pyproof"), attr=cst.Name("require")),
            args=[
                cst.Arg(value=cst.SimpleString(f'"{escaped_claim}"')),
                cst.Arg(value=test_expr)
            ]
        )

        # Return an Expr node (BaseSmallStatement) instead of SimpleStatementLine
        return cst.Expr(value=require_call)

    @staticmethod
    def _expr_to_string(expr: cst.BaseExpression) -> str:
        """Convert expression to string representation"""
        try:
            # Create a temporary module to get the code
            temp_module = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(expr)])])
            code = temp_module.code.strip()
            # Remove the trailing newline and any whitespace
            return code.replace('\n', '').strip()
        except Exception:
            return "complex_expression"

    # ---------------------------------------------------------
    # 3. Loop instrumentation - Enhanced
    # ---------------------------------------------------------
    def leave_While(self, original_node: cst.While, updated: cst.While) -> cst.CSTNode:
        if not self.loops:
            return updated

        node_scope = self.get_metadata(ScopeProvider, original_node)
        if node_scope is None or not isinstance(node_scope, FunctionScope):
            return updated

        if self._has_immediate_break_or_continue(updated.body):
            return updated

        wrapped_body = self._wrap_body_in_proof_context(updated.body, "while_loop_invariant")
        return updated.with_changes(body=wrapped_body)

    def leave_For(self, original_node: cst.For, updated: cst.For) -> cst.CSTNode:
        if not self.loops:
            return updated

        node_scope = self.get_metadata(ScopeProvider, original_node)
        if node_scope is None or not isinstance(node_scope, FunctionScope):
            return updated

        if self._has_immediate_break_or_continue(updated.body):
            return updated

        wrapped_body = self._wrap_body_in_proof_context(updated.body, "for_loop_invariant")
        return updated.with_changes(body=wrapped_body)

    # ---------------------------------------------------------
    # 4. Exception handling instrumentation
    # ---------------------------------------------------------
    def leave_Try(self, original_node: cst.Try, updated: cst.Try) -> cst.Try:
        """Add verification around try/except blocks"""
        if not self.temporal:
            return updated

        # Add temporal events for exception handling
        instrumented_body = self._add_temporal_to_suite(updated.body, "try_block")

        # Instrument exception handlers
        new_handlers = []
        for handler in updated.handlers:
            handler_name = "except_block"
            if handler.type:
                handler_name = f"except_{self._expr_to_string(handler.type)}"
            instrumented_handler_body = self._add_temporal_to_suite(
                handler.body, handler_name
            )
            new_handlers.append(handler.with_changes(body=instrumented_handler_body))

        return updated.with_changes(body=instrumented_body, handlers=new_handlers)

    # ---------------------------------------------------------
    # 5. Information flow tracking - RE-ENABLED with SimpleStatementLine approach
    # ---------------------------------------------------------
    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated: cst.SimpleStatementLine) -> \
    Union[cst.SimpleStatementLine, cst.FlattenSentinel[cst.BaseStatement]]:
        """Add information flow tracking for sensitive assignments at the statement level"""
        if not self.ghost:
            return updated

        # Check if this line contains a sensitive assignment
        for stmt in updated.body:
            if isinstance(stmt, cst.Assign) and self._is_sensitive_assignment(stmt):
                # Create tracking statement
                tracking_stmt = self._create_tainted_statement_line(stmt)
                return cst.FlattenSentinel([updated, tracking_stmt])

        return updated

    # ---------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------
    @staticmethod
    def _wrap_body_in_proof_context(body: cst.BaseSuite, context_name: str) -> cst.BaseSuite:
        """Wrap the body in a proof_context with statement"""
        invariant_call = cst.parse_expression(f"pyproof.proof_context('{context_name}')")

        with_stmt = cst.With(
            items=[cst.WithItem(item=invariant_call)],
            body=body,
        )

        return cst.IndentedBlock(body=[with_stmt])

    def _has_immediate_break_or_continue(self, body: cst.BaseSuite) -> bool:
        """Check if body contains break/continue at the immediate level"""

        class BreakContinueChecker(cst.CSTVisitor):
            def __init__(self):
                super().__init__()
                self.found = False
                self.depth = 0

            def visit_While(self, node: cst.While) -> None:
                self.depth += 1

            def leave_While(self, node: cst.While) -> None:
                self.depth -= 1

            def visit_For(self, node: cst.For) -> None:
                self.depth += 1

            def leave_For(self, node: cst.For) -> None:
                self.depth -= 1

            def visit_Break(self, node: cst.Break) -> None:
                if self.depth == 0:
                    self.found = True

            def visit_Continue(self, node: cst.Continue) -> None:
                if self.depth == 0:
                    self.found = True

        checker = BreakContinueChecker()
        body.visit(checker)
        return checker.found

    @staticmethod
    def _is_file_like_protocol(methods: List[str]) -> bool:
        """Detect file-like protocol"""
        file_methods = {'open', 'read', 'write', 'close'}
        return len(set(methods) & file_methods) >= 3

    @staticmethod
    def _is_state_machine_protocol(methods: List[str]) -> bool:
        """Detect state machine protocol"""
        state_methods = {'start', 'stop', 'reset', 'init', 'initialize', 'process'}
        return len(set(methods) & state_methods) >= 2

    def _create_protocol_decorator(self, method_name: str) -> Optional[cst.Decorator]:
        """Create protocol method decorator"""

        # Map method names to their target states
        file_method_to_state = {
            'open': 'open',
            'read': 'read',
            'write': 'write',
            'close': 'close'
        }

        state_method_to_state = {
            'start': 'running',  # start() transitions to running state
            'stop': 'stopped',  # stop() transitions to stopped state
            'reset': 'stopped',  # reset() transitions to stopped state
            'process': 'process',  # process() transitions to process state
            'initialize': 'stopped'  # initialize() transitions to stopped state
        }

        if self.current_class and method_name in file_method_to_state:
            target_state = file_method_to_state[method_name]
            return cst.Decorator(
                decorator=cst.Call(
                    func=cst.Attribute(value=cst.Name("pyproof"), attr=cst.Name("protocol_method")),
                    args=[
                        cst.Arg(value=cst.Attribute(value=cst.Name("pyproof"), attr=cst.Name("filemanager_protocol"))),
                        cst.Arg(value=cst.SimpleString(f'"{target_state}"'))
                    ]
                )
            )
        elif self.current_class and method_name in state_method_to_state:
            target_state = state_method_to_state[method_name]
            return cst.Decorator(
                decorator=cst.Call(
                    func=cst.Attribute(value=cst.Name("pyproof"), attr=cst.Name("protocol_method")),
                    args=[
                        cst.Arg(value=cst.Attribute(value=cst.Name("pyproof"), attr=cst.Name("statemachine_protocol"))),
                        cst.Arg(value=cst.SimpleString(f'"{target_state}"'))
                    ]
                )
            )
        return None

    @staticmethod
    def _create_ghost_init_statement(func_name: str) -> cst.SimpleStatementLine:
        """Create ghost state initialization"""
        return cst.SimpleStatementLine(
            body=[cst.Expr(value=cst.Call(
                func=cst.Attribute(value=cst.Name("_ghost"), attr=cst.Name("set")),
                args=[
                    cst.Arg(value=cst.SimpleString(f'"{func_name}_entry"')),
                    cst.Arg(value=cst.Name("True"))
                ]
            ))]
        )

    @staticmethod
    def _create_temporal_event_statement(event_name: str) -> cst.SimpleStatementLine:
        """Create temporal event recording statement"""
        return cst.SimpleStatementLine(
            body=[cst.Expr(value=cst.Call(
                func=cst.Attribute(value=cst.Name("pyproof"), attr=cst.Name("record_temporal_event")),
                args=[cst.Arg(value=cst.SimpleString(f'"{event_name}"'))]
            ))]
        )

    def _add_temporal_to_suite(self, suite: cst.BaseSuite, event_prefix: str) -> cst.BaseSuite:
        """Add temporal recording to a suite of statements"""
        if isinstance(suite, cst.IndentedBlock):
            statements = list(suite.body)
            temporal_stmt = self._create_temporal_event_statement(f"{event_prefix}_entered")
            return cst.IndentedBlock(body=[temporal_stmt] + statements)
        return suite

    def _is_sensitive_assignment(self, assign: cst.Assign) -> bool:
        """Check if assignment involves sensitive data"""
        value_str = self._expr_to_string(assign.value).lower()
        return any(pattern in value_str for pattern in self.sensitive_patterns)

    def _create_tainted_statement_line(self, assign: cst.Assign) -> cst.SimpleStatementLine:
        """Create statement line for tainted value tracking"""
        if assign.targets:
            target = assign.targets[0].target
            target_name = self._expr_to_string(target)
            tracking_call = cst.Call(
                func=cst.Attribute(value=cst.Name("pyproof"), attr=cst.Name("track_sensitive_data")),
                args=[
                    cst.Arg(value=cst.SimpleString(f'"{target_name}"')),
                    cst.Arg(value=target)
                ]
            )
            return cst.SimpleStatementLine(body=[cst.Expr(value=tracking_call)])

        return cst.SimpleStatementLine(body=[cst.Pass()])


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dst", type=click.Path(file_okay=False))
@click.option("--loops/--no-loops", default=False, help="Instrument loops with invariants")
@click.option("--ghost/--no-ghost", default=False, help="Add ghost-state scaffolding")
@click.option("--asserts/--no-asserts", default=False, help="Convert assert statements to require() calls")
@click.option("--temporal/--no-temporal", default=False, help="Add temporal event recording")
@click.option("--protocols/--no-protocols", default=False, help="Detect and instrument protocols")
@click.option("--contracts/--no-contracts", default=True, help="Add auto-contract decorators")
@click.option("--all", "enable_all", is_flag=True, help="Enable all instrumentation features")
def cli(src, dst, loops, ghost, asserts, temporal, protocols, contracts, enable_all):
    """
    PyProofify: Automatically instrument Python code with PyProof verification

    Features:
    - Loops: Wrap loops in proof contexts for invariant checking
    - Ghost: Add ghost state tracking at function boundaries
    - Asserts: Convert assert statements to require() calls
    - Temporal: Record temporal events for property verification
    - Protocols: Detect and instrument common usage protocols
    - Contracts: Add auto-contract decorators from type hints
    - All: Enable all instrumentation features
    """
    if enable_all:
        loops = ghost = asserts = temporal = protocols = contracts = True

    src_path = Path(src)
    dst_path = Path(dst)

    # Clean destination
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True)

    # Copy non-Python files verbatim
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.py"))

    # Transform each Python file
    total_files = 0
    transformed_files = 0

    for py_file in src_path.rglob("*.py"):
        rel = py_file.relative_to(src_path)
        target = dst_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        total_files += 1

        try:
            # Parse and transform with metadata
            module = cst.parse_module(py_file.read_text())
            wrapper = metadata.MetadataWrapper(module)
            injector = ContractInjector(
                loops=loops,
                ghost=ghost,
                asserts=asserts,
                temporal=temporal,
                protocols=protocols,
                contracts=contracts
            )
            new_module = wrapper.visit(injector)

            # Prepare header with imports
            header_parts = ["import pyproof"]

            import_parts = ["require"]
            if contracts:
                import_parts.append("auto_contract")
            if protocols:
                import_parts.extend(["protocol_method", "filemanager_protocol", "statemachine_protocol"])
            if temporal:
                import_parts.append("record_temporal_event")
            if ghost:
                import_parts.append("track_sensitive_data")

            header_parts.append(f"from pyproof import {', '.join(import_parts)}")

            # Add refinement types
            header_parts.append("from pyproof import PositiveInt, NonEmptyList")

            if ghost:
                header_parts.append("_ghost = pyproof._ghost")

            header = "\n".join(header_parts) + "\n\n"
            new_source = header + new_module.code

            # Write transformed code
            target.write_text(new_source)
            transformed_files += 1

        except Exception as e:
            click.echo(f"Error processing {py_file}: {e}", err=True)
            # Copy original file if transformation fails
            shutil.copy2(py_file, target)

    # Report results
    features_enabled = []
    if loops: features_enabled.append("loops")
    if ghost: features_enabled.append("ghost")
    if asserts: features_enabled.append("asserts")
    if temporal: features_enabled.append("temporal")
    if protocols: features_enabled.append("protocols")
    if contracts: features_enabled.append("contracts")

    click.echo(f"Proofed project written to {dst_path}")
    click.echo(f"Processed {transformed_files}/{total_files} Python files")
    click.echo(f"Enabled features: {', '.join(features_enabled) if features_enabled else 'none'}")


if __name__ == "__main__":
    cli()