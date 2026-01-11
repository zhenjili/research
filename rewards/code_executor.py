"""
Safe code execution module using Docker containers or subprocess.

Provides isolated execution environment for evaluating generated code
against test cases. Uses Docker for sandboxing with resource limits.

Usage:
    executor = CodeExecutor()
    result = executor.execute_mbpp_style(code, test_cases)
"""

import os
import json
import time
import tempfile
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


@dataclass
class ExecutionResult:
    """Result of code execution."""
    passed: bool
    num_passed: int
    num_total: int
    outputs: List[str]
    errors: List[str]
    execution_time: float
    status: str  # "pass", "fail", "timeout", "error", "compile_error"


class CodeExecutor:
    """
    Safe code execution using Docker containers.

    Features:
    - Resource limits (CPU, memory, time)
    - Network isolation
    - Filesystem isolation
    - Support for MBPP-style and stdin/stdout test cases
    """

    def __init__(
        self,
        docker_image: str = "code-sandbox:latest",
        timeout: int = 10,
        memory_limit: str = "256m",
        cpu_limit: float = 1.0,
        network_disabled: bool = True,
    ):
        self.docker_image = docker_image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self.client = None

        # Try to initialize Docker client
        self._init_docker()

    def _init_docker(self):
        """Initialize Docker client if available."""
        try:
            import docker
            self.client = docker.from_env()
            # Check if image exists
            try:
                self.client.images.get(self.docker_image)
            except docker.errors.ImageNotFound:
                # Try python:3.11-slim as fallback
                self.docker_image = "python:3.11-slim"
                try:
                    self.client.images.get(self.docker_image)
                except docker.errors.ImageNotFound:
                    print(f"Pulling Docker image: {self.docker_image}")
                    self.client.images.pull(self.docker_image)
        except Exception as e:
            print(f"Warning: Docker not available ({e}), using subprocess executor")
            self.client = None

    def execute_mbpp_style(
        self,
        code: str,
        test_cases: List[str],
        entry_point: Optional[str] = None,
        setup_code: str = "",
    ) -> ExecutionResult:
        """
        Execute code with MBPP-style test cases (assert statements).

        Args:
            code: The generated Python code
            test_cases: List of assert statements (e.g., ["assert func(1) == 2"])
            entry_point: Function name to test (optional)
            setup_code: Setup code to run before tests

        Returns:
            ExecutionResult with pass/fail status
        """
        if self.client is None:
            # Fallback to subprocess
            return self._execute_subprocess(code, test_cases, setup_code)

        # Build test script
        full_code = self._build_mbpp_test_script(code, test_cases, setup_code)

        return self._run_in_container(full_code)

    def execute_stdin_stdout(
        self,
        code: str,
        test_inputs: List[str],
        expected_outputs: List[str],
    ) -> ExecutionResult:
        """
        Execute code with stdin/stdout test cases (competitive programming style).

        Args:
            code: The generated Python code (reads from stdin, writes to stdout)
            test_inputs: List of input strings
            expected_outputs: List of expected output strings

        Returns:
            ExecutionResult with pass/fail status
        """
        results = []
        outputs = []
        errors = []
        total_time = 0

        for inp, expected in zip(test_inputs, expected_outputs):
            result = self._run_with_stdin(code, inp)
            total_time += result.execution_time

            if result.status == "pass":
                actual = result.outputs[0].strip() if result.outputs else ""
                expected_clean = expected.strip()

                if actual == expected_clean:
                    results.append(True)
                    outputs.append(actual)
                else:
                    results.append(False)
                    errors.append(f"Expected: {expected_clean[:100]}, Got: {actual[:100]}")
            else:
                results.append(False)
                errors.extend(result.errors)

        return ExecutionResult(
            passed=all(results) if results else False,
            num_passed=sum(results),
            num_total=len(results),
            outputs=outputs,
            errors=errors,
            execution_time=total_time,
            status="pass" if all(results) else "fail"
        )

    def _build_mbpp_test_script(self, code: str, test_cases: List[str], setup_code: str) -> str:
        """Build a test script for MBPP-style tests."""
        script = f'''
{setup_code}

{code}

# Run test cases
_test_results = []
_test_errors = []

'''
        for i, test in enumerate(test_cases):
            # Sanitize test case
            test = test.strip()
            if not test:
                continue
            script += f'''
try:
    {test}
    _test_results.append(True)
except AssertionError as e:
    _test_results.append(False)
    _test_errors.append(f"Test {i}: AssertionError - {{str(e)}}")
except Exception as e:
    _test_results.append(False)
    _test_errors.append(f"Test {i}: {{type(e).__name__}} - {{str(e)}}")
'''

        script += '''
# Print results as JSON
import json
print("__RESULT__" + json.dumps({
    "passed": all(_test_results) if _test_results else False,
    "num_passed": sum(_test_results),
    "num_total": len(_test_results),
    "errors": _test_errors
}))
'''
        return script

    def _run_in_container(self, code: str) -> ExecutionResult:
        """Run code in Docker container."""
        start_time = time.time()

        try:
            # Create container with resource limits
            container = self.client.containers.run(
                self.docker_image,
                command=["python", "-c", code],
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=int(100000 * self.cpu_limit),
                network_disabled=self.network_disabled,
                detach=True,
                remove=False,
            )

            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.timeout)
                exit_code = result["StatusCode"]
                logs = container.logs().decode("utf-8", errors="replace")
            except Exception:
                container.kill()
                container.remove(force=True)
                return ExecutionResult(
                    passed=False,
                    num_passed=0,
                    num_total=1,
                    outputs=[],
                    errors=["Execution timeout"],
                    execution_time=self.timeout,
                    status="timeout"
                )
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

            execution_time = time.time() - start_time

            # Parse results
            if "__RESULT__" in logs:
                result_json = logs.split("__RESULT__")[1].strip()
                try:
                    result_data = json.loads(result_json)
                    return ExecutionResult(
                        passed=result_data["passed"],
                        num_passed=result_data["num_passed"],
                        num_total=result_data["num_total"],
                        outputs=[logs],
                        errors=result_data.get("errors", []),
                        execution_time=execution_time,
                        status="pass" if result_data["passed"] else "fail"
                    )
                except json.JSONDecodeError:
                    pass

            # Execution error
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=1,
                outputs=[logs],
                errors=[f"Exit code: {exit_code}"],
                execution_time=execution_time,
                status="error"
            )

        except Exception as e:
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=1,
                outputs=[],
                errors=[str(e)],
                execution_time=time.time() - start_time,
                status="error"
            )

    def _run_with_stdin(self, code: str, stdin_input: str) -> ExecutionResult:
        """Run code with stdin input using Docker."""
        if self.client is None:
            return self._execute_stdin_subprocess(code, stdin_input)

        start_time = time.time()

        try:
            # Create temp files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_path = f.name

            try:
                container = self.client.containers.run(
                    self.docker_image,
                    command=f"sh -c 'echo \"{stdin_input}\" | python /tmp/solution.py'",
                    volumes={
                        code_path: {'bind': '/tmp/solution.py', 'mode': 'ro'},
                    },
                    mem_limit=self.memory_limit,
                    network_disabled=self.network_disabled,
                    detach=True,
                )

                result = container.wait(timeout=self.timeout)
                logs = container.logs().decode("utf-8", errors="replace")
                container.remove(force=True)

                return ExecutionResult(
                    passed=True,
                    num_passed=1,
                    num_total=1,
                    outputs=[logs],
                    errors=[],
                    execution_time=time.time() - start_time,
                    status="pass"
                )
            finally:
                os.unlink(code_path)

        except Exception as e:
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=1,
                outputs=[],
                errors=[str(e)],
                execution_time=time.time() - start_time,
                status="error"
            )

    def _execute_subprocess(self, code: str, test_cases: List[str], setup_code: str) -> ExecutionResult:
        """Fallback: execute with subprocess (less secure)."""
        full_code = self._build_mbpp_test_script(code, test_cases, setup_code)

        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", "-c", full_code],
                capture_output=True,
                timeout=self.timeout,
                text=True
            )

            execution_time = time.time() - start_time
            output = result.stdout + result.stderr

            if "__RESULT__" in output:
                result_json = output.split("__RESULT__")[1].strip()
                try:
                    result_data = json.loads(result_json)
                    return ExecutionResult(
                        passed=result_data["passed"],
                        num_passed=result_data["num_passed"],
                        num_total=result_data["num_total"],
                        outputs=[output],
                        errors=result_data.get("errors", []),
                        execution_time=execution_time,
                        status="pass" if result_data["passed"] else "fail"
                    )
                except json.JSONDecodeError:
                    pass

            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=len(test_cases),
                outputs=[output],
                errors=[result.stderr[:500] if result.stderr else "Unknown error"],
                execution_time=execution_time,
                status="error"
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=len(test_cases),
                outputs=[],
                errors=["Timeout"],
                execution_time=self.timeout,
                status="timeout"
            )
        except Exception as e:
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=len(test_cases),
                outputs=[],
                errors=[str(e)],
                execution_time=time.time() - start_time,
                status="error"
            )

    def _execute_stdin_subprocess(self, code: str, stdin_input: str) -> ExecutionResult:
        """Execute stdin/stdout test with subprocess."""
        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", "-c", code],
                input=stdin_input,
                capture_output=True,
                timeout=self.timeout,
                text=True
            )

            return ExecutionResult(
                passed=result.returncode == 0,
                num_passed=1 if result.returncode == 0 else 0,
                num_total=1,
                outputs=[result.stdout],
                errors=[result.stderr] if result.stderr else [],
                execution_time=time.time() - start_time,
                status="pass" if result.returncode == 0 else "fail"
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=1,
                outputs=[],
                errors=["Timeout"],
                execution_time=self.timeout,
                status="timeout"
            )
        except Exception as e:
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=1,
                outputs=[],
                errors=[str(e)],
                execution_time=time.time() - start_time,
                status="error"
            )


class SubprocessExecutor:
    """
    Simpler executor using subprocess (less secure, for development).

    Use Docker executor in production.
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def execute_mbpp_style(
        self,
        code: str,
        test_cases: List[str],
        entry_point: Optional[str] = None,
        setup_code: str = "",
    ) -> ExecutionResult:
        """Execute with subprocess (development only)."""
        full_code = f"{setup_code}\n{code}\n"

        passed = []
        errors = []

        for test in test_cases:
            if not test.strip():
                continue
            test_code = full_code + test
            try:
                result = subprocess.run(
                    ["python", "-c", test_code],
                    capture_output=True,
                    timeout=self.timeout,
                    text=True
                )
                if result.returncode == 0:
                    passed.append(True)
                else:
                    passed.append(False)
                    errors.append(result.stderr[:200])
            except subprocess.TimeoutExpired:
                passed.append(False)
                errors.append("Timeout")
            except Exception as e:
                passed.append(False)
                errors.append(str(e))

        return ExecutionResult(
            passed=all(passed) if passed else False,
            num_passed=sum(passed),
            num_total=len(passed),
            outputs=[],
            errors=errors,
            execution_time=0,
            status="pass" if all(passed) else "fail"
        )


# Test the executor
if __name__ == "__main__":
    print("Testing CodeExecutor...")

    executor = CodeExecutor(timeout=5)

    # Test 1: Simple passing test
    code1 = """
def add(a, b):
    return a + b
"""
    tests1 = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    result1 = executor.execute_mbpp_style(code1, tests1)
    print(f"Test 1 (should pass): {result1.status}, passed={result1.passed}")

    # Test 2: Failing test
    code2 = """
def add(a, b):
    return a - b  # Bug!
"""
    tests2 = ["assert add(1, 2) == 3"]
    result2 = executor.execute_mbpp_style(code2, tests2)
    print(f"Test 2 (should fail): {result2.status}, passed={result2.passed}")

    # Test 3: Timeout
    code3 = """
import time
time.sleep(10)
"""
    tests3 = ["assert True"]
    result3 = executor.execute_mbpp_style(code3, tests3)
    print(f"Test 3 (should timeout): {result3.status}")

    print("\nAll tests completed!")
