r"""Offline tests for the invoice classifier, its assets, and Gradio wiring.

Run from the project directory on Windows::

    .venv\Scripts\python.exe tests\tests.py -v

Or use unittest discovery (no additional test dependencies are required)::

    python -m unittest discover -s tests -p 'test*.py' -v

The application dependencies in requirements.txt must be installed. External
processes, HTTP requests, sleeps, and UI launches are blocked or mocked. The
real Pydantic models, JSONL assets, Gradio components, and in-process
pydantic-ai models are exercised. No tests are skipped or marked as expected
failures. This suite does not assess real LLM classification quality, GPU/model
startup, or browser interaction.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError, URLError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, RetryPromptPart, SystemPromptPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

import interface
import llm_req as llm
import main as app


def category_json(*names):
    return json.dumps({"Categories": list(names)}, ensure_ascii=False)


def konto_json(*numbers):
    return json.dumps({"Kontos": [
        {"Konto": number, "data": [number, "Bürobedarf"]}
        for number in numbers
    ]}, ensure_ascii=False)


class SafeTestCase(unittest.TestCase):
    """Fail immediately if a test forgets to replace an external operation."""

    def setUp(self):
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(patch("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False))
        self.popen = self.enterContext(patch.object(
            llm.subprocess, "Popen", side_effect=AssertionError("Unexpected process launch")
        ))
        self.run = self.enterContext(patch.object(
            llm.subprocess, "run", side_effect=AssertionError("Unexpected subprocess.run")
        ))
        self.urlopen = self.enterContext(patch.object(
            llm, "urlopen", side_effect=AssertionError("Unexpected HTTP request")
        ))
        self.sleep = self.enterContext(patch.object(llm.time, "sleep"))
        self.launch = self.enterContext(patch.object(
            app.gr.Blocks, "launch", autospec=True,
            side_effect=AssertionError("Unexpected UI launch"),
        ))

    def allow_process(self, *, returncode=0, stdout="answer", stderr=""):
        process = Mock(pid=1234, returncode=returncode)
        process.communicate.return_value = (stdout, stderr)
        self.popen.side_effect = None
        self.popen.return_value = process
        return process

    def healthy_response(self, status=200):
        response = MagicMock()
        response.__enter__.return_value.status = status
        return response

    def temporary_directory(self):
        return Path(self.enterContext(tempfile.TemporaryDirectory()))


class SchemaTests(SafeTestCase):
    def test_category_preserves_relevance_order_and_unicode(self):
        names = ["Anlagevermögen", "Eigenkapital"]
        profile = llm.UserProfile_Category(Categories=names)
        self.assertEqual(json.loads(profile.model_dump_json()), {"Categories": names})

    def test_category_rejects_missing_or_invalid_fields(self):
        for payload in ({}, {"Categories": "Eigenkapital"}, {"Categories": [1]},
                        {"Categories": None}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                llm.UserProfile_Category.model_validate(payload)

    def test_account_preserves_leading_zeros_and_data(self):
        payload = {"Konto": "00100", "data": ["00100", "Übernachtung", "VSt. 19%"]}
        account = llm.Konto.model_validate(payload)
        self.assertEqual(json.loads(account.model_dump_json()), payload)

    def test_account_rejects_incomplete_or_invalid_data(self):
        for payload in ({"Konto": "00100"}, {"data": []},
                        {"Konto": 100, "data": []},
                        {"Konto": "00100", "data": "text"},
                        {"Konto": "00100", "data": [1]}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                llm.Konto.model_validate(payload)

    def test_accounts_parse_nested_json(self):
        profile = llm.UserProfile_Kontos.model_validate_json(konto_json("00100", "06000"))
        self.assertTrue(all(isinstance(row, llm.Konto) for row in profile.Kontos))
        self.assertEqual([row.Konto for row in profile.Kontos], ["00100", "06000"])

    def test_accounts_allow_no_matches(self):
        self.assertEqual(llm.UserProfile_Kontos.model_validate_json(konto_json()).Kontos, [])

    def test_accounts_reject_invalid_nested_objects(self):
        for payload in ({}, {"Kontos": None}, {"Kontos": [{"Konto": "00100"}]}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                llm.UserProfile_Kontos.model_validate(payload)

    def test_empty_dict_answer_schema(self):
        self.assertEqual(llm.Empty_dict(Answer=["Описание чека"]).Answer, ["Описание чека"])
        for payload in ({}, {"Answer": "text"}, {"Answer": [1]}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                llm.Empty_dict.model_validate(payload)


class ProjectAssetsTests(SafeTestCase):
    def test_paths_are_absolute_and_relative_to_module(self):
        self.assertEqual(llm.BASE_DIR, PROJECT_ROOT)
        for path in [llm.sys_prompt_categorie, llm.sys_prompt_konto,
                     *llm.category_files.values()]:
            with self.subTest(path=path):
                self.assertTrue(path.is_absolute())
                self.assertEqual(path.parent, PROJECT_ROOT)
                self.assertTrue(path.is_file())

    def test_category_mapping_covers_all_shipped_jsonl_files(self):
        actual = set(PROJECT_ROOT.glob("Kontenrahmen_categories_*.jsonl"))
        self.assertEqual(set(llm.category_files.values()), actual)
        self.assertEqual(len(llm.category_files), 11)
        for category, path in llm.category_files.items():
            with self.subTest(category=category):
                self.assertEqual(path.name, f"Kontenrahmen_categories_{category}.jsonl")

    def test_prompts_are_nonempty_utf8_and_reference_output(self):
        category_prompt = llm.read_md(llm.sys_prompt_categorie)
        konto_prompt = llm.read_md(llm.sys_prompt_konto)
        self.assertTrue(category_prompt.strip())
        self.assertIn("Rechnung", category_prompt)
        for category in llm.category_files:
            with self.subTest(category=category):
                self.assertIn(f'"{category}"', category_prompt)
        for field in ("Kontos", "Konto", "data"):
            self.assertIn(f'"{field}"', konto_prompt)

    def test_every_jsonl_record_matches_account_schema(self):
        total = 0
        for category, path in llm.category_files.items():
            lines = path.read_text(encoding="utf-8").splitlines()
            if category != "Nicht_belegte_Kontenklasse":
                self.assertTrue(lines, f"Unexpected empty category: {category}")
            for number, line in enumerate(lines, 1):
                with self.subTest(file=path.name, line=number):
                    row = json.loads(line)
                    account = llm.Konto.model_validate(row)
                    self.assertEqual(set(row), {"Konto", "data"})
                    self.assertRegex(account.Konto, r"^\d{5}$")
                    self.assertTrue(account.data)
                    self.assertEqual(account.data[0], account.Konto)
                    total += 1
        self.assertGreater(total, 0)

    def test_account_numbers_are_unique_across_all_categories(self):
        seen = {}
        for path in llm.category_files.values():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                account = json.loads(line)["Konto"]
                with self.subTest(file=path.name, line=number, account=account):
                    self.assertNotIn(account, seen, f"Duplicate of {seen.get(account)}")
                    seen[account] = (path.name, number)

    def test_all_categories_can_be_loaded_together(self):
        merged = llm.get_all_kontos(category_json(*llm.category_files))
        expected = [line for path in llm.category_files.values()
                    for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(merged.splitlines(), expected)


class MarkdownTests(SafeTestCase):
    def test_reads_utf8_without_stripping_content(self):
        path = self.temporary_directory() / "промпт.md"
        text = "  # Übernachtung\nОписание чека\n\n"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(llm.read_md(path), text)
        self.assertEqual(llm.read_md(str(path)), text)

    def test_empty_file(self):
        path = self.temporary_directory() / "empty.md"
        path.write_text("", encoding="utf-8")
        self.assertEqual(llm.read_md(path), "")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            llm.read_md(self.temporary_directory() / "missing.md")

    def test_invalid_utf8(self):
        path = self.temporary_directory() / "invalid.md"
        path.write_bytes(b"\xff\xfe\xfa")
        with self.assertRaises(UnicodeDecodeError):
            llm.read_md(path)


class GetAllKontosTests(SafeTestCase):
    def setUp(self):
        super().setUp()
        root = self.temporary_directory()
        self.files = {key: root / f"{key}.jsonl" for key in ("A", "B", "Empty")}
        self.files["A"].write_bytes('{"Konto":"00100","data":["Ä"]}\r\n\r\n'.encode())
        self.files["B"].write_text('{"Konto":"00200","data":[]}\n', encoding="utf-8")
        self.files["Empty"].write_text("", encoding="utf-8")
        self.enterContext(patch.dict(llm.category_files, self.files, clear=True))

    def test_concatenates_in_requested_order_with_one_separator(self):
        expected = '{"Konto":"00200","data":[]}\n{"Konto":"00100","data":["Ä"]}'
        self.assertEqual(llm.get_all_kontos(category_json("B", "A")), expected)

    def test_empty_selection_returns_empty_string(self):
        self.assertEqual(llm.get_all_kontos(category_json()), "")

    def test_empty_categories_do_not_add_blank_lines(self):
        actual = llm.get_all_kontos(category_json("Empty", "A", "Empty", "B", "Empty"))
        self.assertEqual(len(actual.splitlines()), 2)
        self.assertFalse(actual.startswith("\n"))
        self.assertFalse(actual.endswith("\n"))

    def test_only_empty_category(self):
        self.assertEqual(llm.get_all_kontos(category_json("Empty")), "")

    def test_unknown_category(self):
        with self.assertRaises(KeyError):
            llm.get_all_kontos(category_json("Unknown"))

    def test_invalid_json_and_missing_categories(self):
        for payload, error in (("not json", json.JSONDecodeError), ("{}", KeyError)):
            with self.subTest(payload=payload), self.assertRaises(error):
                llm.get_all_kontos(payload)

    def test_missing_category_file(self):
        self.files["A"].unlink()
        with self.assertRaises(FileNotFoundError):
            llm.get_all_kontos(category_json("A"))

    def test_paths_work_from_another_working_directory(self):
        previous_directory = Path.cwd()
        other_directory = self.temporary_directory()
        # Restore cwd before TemporaryDirectory's registered cleanup runs.
        self.addCleanup(os.chdir, previous_directory)
        os.chdir(other_directory)
        self.assertIn('"00100"', llm.get_all_kontos(category_json("A")))


class WaitForServerTests(SafeTestCase):
    def setUp(self):
        super().setUp()
        self.process = Mock(returncode=None)
        self.process.poll.return_value = None
        self.monotonic = self.enterContext(patch.object(llm.time, "monotonic", return_value=0))

    def test_ready_server_returns_without_sleeping(self):
        self.urlopen.side_effect = None
        self.urlopen.return_value = self.healthy_response()
        self.assertIsNone(llm.wait_for_server(self.process))
        self.urlopen.assert_called_once_with("http://127.0.0.1:8080/health", timeout=2.0)
        self.sleep.assert_not_called()

    def test_transient_http_connection_and_timeout_errors_are_retried(self):
        errors = [HTTPError("url", 503, "loading", None, None),
                  URLError("connection refused"), TimeoutError("not ready")]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.urlopen.reset_mock()
                self.sleep.reset_mock()
                self.urlopen.side_effect = [error, self.healthy_response()]
                llm.wait_for_server(self.process)
                self.assertEqual(self.urlopen.call_count, 2)
                self.sleep.assert_called_once_with(0.25)

    def test_non_200_response_is_retried(self):
        self.urlopen.side_effect = [self.healthy_response(202), self.healthy_response()]
        llm.wait_for_server(self.process)
        self.sleep.assert_called_once_with(0.25)

    def test_other_http_errors_are_not_swallowed(self):
        for code in (401, 404, 500):
            with self.subTest(code=code):
                self.urlopen.side_effect = HTTPError("url", code, "error", None, None)
                with self.assertRaises(HTTPError) as error:
                    llm.wait_for_server(self.process)
                self.assertEqual(error.exception.code, code)
        self.sleep.assert_not_called()

    def test_already_exited_process_fails_before_http(self):
        self.process.poll.return_value = 9
        self.process.returncode = 9
        with self.assertRaisesRegex(RuntimeError, "9"):
            llm.wait_for_server(self.process)
        self.urlopen.assert_not_called()

    def test_process_exit_during_loading(self):
        self.process.poll.side_effect = [None, 3]
        self.process.returncode = 3
        self.urlopen.side_effect = URLError("loading")
        with self.assertRaisesRegex(RuntimeError, "3"):
            llm.wait_for_server(self.process)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_zero_and_negative_timeout_do_not_request_http(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout), self.assertRaises(TimeoutError):
                llm.wait_for_server(self.process, timeout=timeout)
        self.urlopen.assert_not_called()

    def test_request_timeout_is_capped_by_remaining_time(self):
        self.monotonic.side_effect = [10.0, 10.75]
        self.urlopen.side_effect = None
        self.urlopen.return_value = self.healthy_response()
        llm.wait_for_server(self.process, timeout=1)
        self.assertEqual(self.urlopen.call_args.kwargs["timeout"], 0.25)

    def test_overall_deadline_expires_after_transient_error(self):
        self.monotonic.side_effect = [0, 0, 1]
        self.urlopen.side_effect = URLError("loading")
        with self.assertRaises(TimeoutError):
            llm.wait_for_server(self.process, timeout=1)
        self.assertEqual(self.urlopen.call_count, 1)


class ServerProcessTests(SafeTestCase):
    def test_missing_model_never_launches_process(self):
        with patch.object(Path, "glob", return_value=iter([])):
            with self.assertRaisesRegex(FileNotFoundError, "PQ2_0"):
                llm.run_llama_server()
        self.popen.assert_not_called()

    def test_server_uses_matching_model_and_windows_launch_flags(self):
        process = self.allow_process()
        base = Path(r"C:\WINDOWS\system32\Bonsai-demo")
        model = base / "models/bonsai2-gguf/27B/model-PQ2_0.gguf"
        with patch.object(Path, "glob", return_value=iter([model])), patch.object(
            llm.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True
        ):
            self.assertIs(llm.run_llama_server(), process)
        command = self.popen.call_args.args[0]
        self.assertEqual(command[0], str(base / "bin/cuda/llama-server.exe"))
        self.assertEqual(command[command.index("-m") + 1], str(model))
        self.assertEqual(command[command.index("--alias") + 1], "bonsai2")
        self.assertEqual(self.popen.call_args.kwargs["cwd"], base)
        self.assertEqual(self.popen.call_args.kwargs["creationflags"], 0x08000000)

    def test_server_launch_failure_is_propagated(self):
        self.popen.side_effect = OSError("cannot start llama-server")
        with patch.object(Path, "glob", return_value=iter([Path("model-PQ2_0.gguf")])), \
             patch.object(llm.subprocess, "CREATE_NO_WINDOW", 0, create=True):
            with self.assertRaisesRegex(OSError, "cannot start"):
                llm.run_llama_server()

    def test_stopping_exited_server_does_not_kill_anything(self):
        process = Mock()
        process.poll.return_value = 0
        llm.stop_server_tree(process)
        self.run.assert_not_called()
        process.wait.assert_not_called()

    def test_stopping_active_server_kills_tree_and_waits(self):
        process = Mock(pid=1234)
        process.poll.return_value = None
        self.run.side_effect = None
        llm.stop_server_tree(process)
        self.run.assert_called_once_with(
            ["taskkill", "/PID", "1234", "/T", "/F"],
            capture_output=True, text=True, check=False,
        )
        process.wait.assert_called_once_with(timeout=5)

    def test_taskkill_nonzero_exit_still_waits_for_parent(self):
        process = Mock(pid=1234)
        process.poll.return_value = None
        self.run.side_effect = None
        self.run.return_value.returncode = 128
        llm.stop_server_tree(process)
        process.wait.assert_called_once_with(timeout=5)

    def test_stop_timeout_is_handled(self):
        process = Mock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("taskkill", 5)
        self.run.side_effect = None
        self.assertIsNone(llm.stop_server_tree(process))
        self.assertIn("Не удалось дождаться", self.output.getvalue())


class LocalModelRequestTests(SafeTestCase):
    def setUp(self):
        super().setUp()
        # A regression in the application's unbounded retry loop must not hang
        # the test runner. The guard raises from sleep, outside the retry try.
        self.sleep.side_effect = [None] * 3 + [AssertionError("Test retry budget exceeded")]
        self.server = self.enterContext(patch.object(llm, "run_llama_server"))
        self.ready = self.enterContext(patch.object(llm, "wait_for_server"))
        self.stop = self.enterContext(patch.object(llm, "stop_server_tree"))
        self.provider = self.enterContext(patch.object(llm, "OpenAIProvider"))
        self.model = self.enterContext(patch.object(llm, "OpenAIChatModel"))

    def request(self, **overrides):
        kwargs = dict(message="Bürobedarf", sys_prompt="Classify",
                      profile=llm.UserProfile_Category, kontos="")
        kwargs.update(overrides)
        return llm.local_model_request(**kwargs)

    def mock_agent(self):
        constructor = self.enterContext(patch.object(llm, "Agent"))
        constructor.return_value.run_sync.return_value.output = llm.UserProfile_Category(
            Categories=["Eigenkapital"]
        )
        return constructor

    def test_structured_output_and_provider_configuration(self):
        agent = self.mock_agent()
        result = self.request(kontos="account records")
        self.assertEqual(json.loads(result), {"Categories": ["Eigenkapital"]})
        self.provider.assert_called_once_with(base_url="http://127.0.0.1:8080/v1", api_key="dummy")
        self.model.assert_called_once_with("local-model", provider=self.provider.return_value)
        agent.assert_called_once_with(
            self.model.return_value, output_type=llm.UserProfile_Category,
            system_prompt="Classify\nBürobedarf\naccount records",
        )
        args, kwargs = agent.return_value.run_sync.call_args
        self.assertEqual(args, ("Bürobedarf",))
        self.assertEqual(kwargs["usage_limits"].request_limit, 3)
        self.ready.assert_called_once_with(self.server.return_value)
        self.stop.assert_called_once_with(self.server.return_value)

    def test_path_prompt_is_read_as_utf8(self):
        self.mock_agent()
        with patch.object(llm, "read_md", return_value="Инструкция") as read:
            self.request(sys_prompt=Path("prompt.md"))
        read.assert_called_once_with(Path("prompt.md"))

    def test_string_prompt_is_not_treated_as_a_filename(self):
        self.mock_agent()
        with patch.object(llm, "read_md") as read:
            self.request(sys_prompt="literal text")
        read.assert_not_called()

    def test_real_pydantic_ai_agent_handles_each_output_schema(self):
        examples = [
            (llm.UserProfile_Category, {"Categories": ["Anlagevermögen"]}),
            (llm.UserProfile_Kontos, json.loads(konto_json("00100"))),
            (llm.Empty_dict, {"Answer": ["Нужно описание"]}),
        ]
        for profile, output in examples:
            with self.subTest(profile=profile.__name__), asyncio.Runner() as runner:
                runner.get_loop()
                self.model.return_value = TestModel(custom_output_args=output)
                self.assertEqual(json.loads(self.request(profile=profile)), output)

    def assert_model_output_is_repaired(self, invalid_args):
        """Inject invalid raw MODEL output, leaving Agent validation intact."""
        requests = []
        expected = json.loads(konto_json("00100"))

        async def respond(messages, info):
            requests.append(list(messages))
            args = invalid_args if len(requests) == 1 else expected
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

        self.model.return_value = FunctionModel(respond)
        with asyncio.Runner() as runner:
            runner.get_loop()
            actual = self.request(profile=llm.UserProfile_Kontos)

        self.assertEqual(json.loads(actual), expected)
        self.assertEqual(len(requests), 2)
        retry_parts = [part for message in requests[1] for part in message.parts
                       if isinstance(part, RetryPromptPart)]
        self.assertTrue(retry_parts, "Agent must request a corrected model response")
        self.sleep.assert_not_called()  # Agent repaired output within one run_sync.
        self.server.assert_called_once()
        self.stop.assert_called_once_with(self.server.return_value)

    def test_real_agent_retries_malformed_json_from_model(self):
        self.assert_model_output_is_repaired("this is not JSON")

    def test_real_agent_retries_missing_kontos_field_from_model(self):
        self.assert_model_output_is_repaired({})

    def test_exhausted_agent_validation_retries_restart_request(self):
        requests = []
        expected = json.loads(konto_json("00100"))

        async def respond(messages, info):
            requests.append(list(messages))
            # Two invalid responses exhaust the Agent's default output retries.
            args = {} if len(requests) <= 2 else expected
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

        self.model.return_value = FunctionModel(respond)
        with asyncio.Runner() as runner:
            runner.get_loop()
            actual = self.request(profile=llm.UserProfile_Kontos)

        self.assertEqual(json.loads(actual), expected)
        self.assertEqual(len(requests), 3)
        self.sleep.assert_called_once_with(2)
        self.server.assert_called_once()
        self.stop.assert_called_once_with(self.server.return_value)

    def test_transient_agent_failure_is_retried(self):
        agent = self.mock_agent().return_value
        result = agent.run_sync.return_value
        agent.run_sync.side_effect = [RuntimeError("temporary failure"), result]
        self.assertEqual(json.loads(self.request()), {"Categories": ["Eigenkapital"]})
        self.assertEqual(agent.run_sync.call_count, 2)
        self.sleep.assert_called_once_with(2)
        self.stop.assert_called_once()

    def test_serialization_failure_is_retried(self):
        agent = self.mock_agent().return_value
        bad_result = Mock()
        bad_result.output.model_dump_json.side_effect = ValueError("bad output")
        agent.run_sync.side_effect = [bad_result, agent.run_sync.return_value]
        self.request()
        self.assertEqual(agent.run_sync.call_count, 2)
        self.sleep.assert_called_once_with(2)

    def test_readiness_failure_stops_server(self):
        self.ready.side_effect = TimeoutError("loading timed out")
        with self.assertRaises(TimeoutError):
            self.request()
        self.stop.assert_called_once_with(self.server.return_value)
        self.model.assert_not_called()

    def test_prompt_read_failure_stops_server(self):
        with patch.object(llm, "read_md", side_effect=FileNotFoundError("prompt")):
            with self.assertRaises(FileNotFoundError):
                self.request(sys_prompt=Path("missing.md"))
        self.stop.assert_called_once_with(self.server.return_value)

    def test_provider_setup_failure_stops_server(self):
        self.provider.side_effect = ValueError("bad provider")
        with self.assertRaisesRegex(ValueError, "bad provider"):
            self.request()
        self.stop.assert_called_once_with(self.server.return_value)

    def test_agent_setup_failure_stops_server(self):
        agent = self.mock_agent()
        agent.side_effect = ValueError("invalid schema")
        with self.assertRaisesRegex(ValueError, "invalid schema"):
            self.request()
        self.stop.assert_called_once_with(self.server.return_value)

    def test_launch_failure_does_not_stop_a_nonexistent_process(self):
        self.server.side_effect = FileNotFoundError("model missing")
        with self.assertRaises(FileNotFoundError):
            self.request()
        self.stop.assert_not_called()

    def test_interruption_exits_retry_loop_and_stops_server(self):
        agent = self.mock_agent().return_value
        agent.run_sync.side_effect = [RuntimeError("temporary failure"), KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.request()
        self.assertEqual(agent.run_sync.call_count, 2)
        self.stop.assert_called_once_with(self.server.return_value)


class CategoryStepTests(SafeTestCase):
    def setUp(self):
        super().setUp()
        self.request = self.enterContext(patch.object(llm, "local_model_request"))
        self.valid = category_json("Eigenkapital", "Anlagevermögen")
        self.request.return_value = self.valid

    def test_success_preserves_response_and_uses_category_schema(self):
        self.assertEqual(llm.step_one_get_category("receipt", "prompt"), self.valid)
        self.request.assert_called_once_with(
            message="receipt", sys_prompt="prompt", profile=llm.UserProfile_Category, kontos=""
        )
        self.sleep.assert_not_called()

    def test_unknown_category_and_request_errors_retry_once(self):
        for invalid in (category_json("Unknown"), RuntimeError("offline")):
            with self.subTest(invalid=invalid):
                self.request.reset_mock()
                self.sleep.reset_mock()
                self.request.side_effect = [invalid, self.valid]
                self.assertEqual(llm.step_one_get_category("receipt", "prompt"), self.valid)
                self.assertEqual(self.request.call_count, 2)
                self.sleep.assert_called_once_with(2)

    def test_persistent_unknown_category_is_rejected(self):
        self.request.return_value = category_json("Unknown")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            llm.step_one_get_category("receipt", "prompt")
        self.assertEqual(self.request.call_count, 2)
        self.sleep.assert_called_once_with(2)

    def test_final_request_error_is_preserved(self):
        first = RuntimeError("first")
        last = RuntimeError("last")
        self.request.side_effect = [first, last]
        with self.assertRaises(RuntimeError) as caught:
            llm.step_one_get_category("receipt", "prompt")
        self.assertIs(caught.exception, last)
        self.sleep.assert_called_once_with(2)


class KontoStepTests(SafeTestCase):
    def setUp(self):
        super().setUp()
        self.request = self.enterContext(patch.object(llm, "local_model_request"))
        self.request.return_value = konto_json("00100", "06000")

    def test_formats_account_numbers_in_relevance_order(self):
        result = llm.step_two_get_konto("receipt", "prompt", category_json("Eigenkapital"), "rows")
        self.assertEqual(result, '"Konto":"00100"\n"Konto":"06000"')
        self.request.assert_called_once_with(
            message="receipt", sys_prompt="prompt", profile=llm.UserProfile_Kontos, kontos="rows"
        )
        self.sleep.assert_not_called()

    def test_no_matching_accounts_returns_empty_string(self):
        self.request.return_value = konto_json()
        self.assertEqual(llm.step_two_get_konto("receipt", "prompt", all_kontos="rows"), "")

    def test_transient_request_failure_retries_once(self):
        self.request.side_effect = [RuntimeError("offline"), konto_json("00100")]
        self.assertEqual(llm.step_two_get_konto("receipt", "prompt", all_kontos="rows"),
                         '"Konto":"00100"')
        self.assertEqual(self.request.call_count, 2)
        self.sleep.assert_called_once_with(2)

    def test_persistent_request_failure_is_propagated(self):
        error = RuntimeError("offline")
        self.request.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            llm.step_two_get_konto("receipt", "prompt", all_kontos="rows")
        self.assertIs(caught.exception, error)
        self.assertEqual(self.request.call_count, 2)
        self.sleep.assert_called_once_with(2)


class MainWorkflowTests(SafeTestCase):
    def setUp(self):
        super().setUp()
        self.category = self.enterContext(patch.object(llm, "step_one_get_category"))
        self.accounts = self.enterContext(patch.object(llm, "get_all_kontos"))
        self.select = self.enterContext(patch.object(llm, "step_two_get_konto"))
        self.category.return_value = category_json("Eigenkapital")
        self.accounts.return_value = "row one\nrow two\n"
        self.select.return_value = '"Konto":"02000"'
        self.prompts = [Path("category.md"), Path("konto.md"), None]

    def request(self, message="Rechnung Bürobedarf", **kwargs):
        return llm.main(message, [], self.prompts, **{"model_type": "local", **kwargs})

    def test_local_workflow_passes_cleaned_message_and_correct_prompts(self):
        self.assertEqual(self.request().splitlines(), ['"Konto":"02000"'])
        self.category.assert_called_once_with("Bürobedarf", self.prompts[0])
        self.accounts.assert_called_once_with(self.category.return_value)
        self.select.assert_called_once_with(
            "Bürobedarf", self.prompts[1], self.category.return_value, self.accounts.return_value
        )

    def test_missing_keyword_requests_description_without_model_calls(self):
        for message in ("", "Hotelübernachtung", "1234", "\n\t", "Abrechnung Hotel", "Rechnungen Hotel"):
            with self.subTest(message=message):
                self.assertIn("Введите 'Rechnung'", self.request(message))
        self.category.assert_not_called()
        self.select.assert_not_called()

    def test_keyword_without_description_is_rejected(self):
        for message in ("Rechnung", "rechnung", "Rechnung: 123.45 €", "Rechnung -- ___", "Rechnung rechnung"):
            with self.subTest(message=message):
                self.assertIn("с полным описанием", self.request(message))
        self.category.assert_not_called()
        self.select.assert_not_called()

    def test_lowercase_keyword_and_non_latin_description_are_supported(self):
        for message, cleaned in (("rechnung Hotel", "Hotel"), ("Rechnung Канцелярия", "Канцелярия")):
            with self.subTest(message=message):
                self.category.reset_mock()
                self.request(message)
                self.category.assert_called_once_with(cleaned, self.prompts[0])

    def test_uppercase_and_mixed_case_keywords_are_supported(self):
        for keyword in ("RECHNUNG", "ReChNuNg"):
            with self.subTest(keyword=keyword):
                self.category.reset_mock()
                self.assertEqual(self.request(f"{keyword} Hotel").splitlines(),
                                 [self.select.return_value])
                self.category.assert_called_once_with("Hotel", self.prompts[0])

    def test_description_after_punctuation_or_leading_whitespace_is_accepted(self):
        for message in ("Rechnung: Hotel", "  Rechnung: Hotel", "\trechnung -- Hotel"):
            with self.subTest(message=message):
                self.category.reset_mock()
                self.assertEqual(self.request(message).splitlines(), [self.select.return_value])
                self.category.assert_called_once()
                self.assertIn("Hotel", self.category.call_args.args[0])

    def test_keyword_must_be_at_the_beginning_as_interface_instructs(self):
        result = self.request("Hotel Rechnung Frühstück")
        self.assertIn("в начале", result)
        self.category.assert_not_called()

    def test_exact_batch_limit_does_not_create_an_extra_request(self):
        self.accounts.return_value = "a" * 14999 + "\n" + "b" * 14999 + "\n"
        self.request()
        self.select.assert_called_once()
        self.assertEqual(self.select.call_args.args[3], self.accounts.return_value)

    def test_batching_preserves_all_lines_and_respects_limit(self):
        self.accounts.return_value = "a" * 14999 + "\n" + "b" * 14999 + "\n" + "c\n"
        self.request()
        batches = [item.args[3] for item in self.select.call_args_list]
        self.assertEqual([len(batch) for batch in batches], [30000, 2])
        self.assertEqual("".join(batches), self.accounts.return_value)
        self.assertTrue(all(batch.endswith("\n") for batch in batches))

    def test_single_oversized_jsonl_record_is_not_split(self):
        large_record = "a" * 30001 + "\n"
        self.accounts.return_value = large_record + "b\n"
        self.request()
        self.assertEqual([item.args[3] for item in self.select.call_args_list], [large_record, "b\n"])

    def test_results_from_multiple_batches_have_a_line_separator(self):
        self.accounts.return_value = "a" * 29999 + "\n" + "b\n"
        self.select.side_effect = ['"Konto":"00100"', '"Konto":"00200"']
        # splitlines accepts the final newline while still detecting glued rows
        # or an unintended blank line between the two responses.
        self.assertEqual(self.request().splitlines(), ['"Konto":"00100"', '"Konto":"00200"'])

    def test_batching_preserves_final_record_without_a_trailing_newline(self):
        self.accounts.return_value = "a" * 29999 + "\n" + "final record"
        self.request()
        self.assertEqual([item.args[3] for item in self.select.call_args_list],
                         ["a" * 29999 + "\n", "final record"])

    def test_batches_without_matching_accounts_return_no_account_text(self):
        self.select.return_value = ""
        self.assertEqual(self.request().strip(), "")
        self.select.assert_called_once()

    def test_empty_account_list_does_not_send_empty_batch_to_model(self):
        self.accounts.return_value = ""
        self.assertEqual(self.request(), "")
        self.select.assert_not_called()

    def test_category_errors_stop_workflow(self):
        self.category.side_effect = ValueError("invalid category")
        with self.assertRaisesRegex(ValueError, "invalid category"):
            self.request()
        self.accounts.assert_not_called()
        self.select.assert_not_called()

    def test_account_file_errors_stop_workflow(self):
        self.accounts.side_effect = FileNotFoundError("missing JSONL")
        with self.assertRaises(FileNotFoundError):
            self.request()
        self.select.assert_not_called()

    def test_account_selection_errors_are_propagated(self):
        self.select.side_effect = RuntimeError("model failed")
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            self.request()

    def test_default_backend_is_local(self):
        result = llm.main("Rechnung Hotel", [], self.prompts)
        self.assertEqual(result.splitlines(), [self.select.return_value])
        self.category.assert_called_once_with("Hotel", self.prompts[0])
        self.select.assert_called_once()

    def test_unsupported_backend_raises_clear_value_error(self):
        for model_type in ("unsupported", "codex", ""):
            with self.subTest(model_type=model_type), self.assertRaisesRegex(
                ValueError, "(?i)model|модел|backend"
            ):
                self.request(model_type=model_type)
        self.category.assert_not_called()
        self.accounts.assert_not_called()
        self.select.assert_not_called()


class OfflinePipelineTests(SafeTestCase):
    """Exercise the actual callbacks, both steps, prompts, files, and Agent."""

    def setUp(self):
        super().setUp()
        self.server = self.enterContext(patch.object(llm, "run_llama_server"))
        self.ready = self.enterContext(patch.object(llm, "wait_for_server"))
        self.stop = self.enterContext(patch.object(llm, "stop_server_tree"))
        self.enterContext(patch.object(llm, "OpenAIProvider"))
        self.model = self.enterContext(patch.object(llm, "OpenAIChatModel"))
        self.sleep.side_effect = AssertionError("Valid pipeline unexpectedly retried")

    def run_pipeline(self, category):
        description = "Testbeleg Bürobedarf"
        source = llm.category_files[category].read_text(encoding="utf-8").rstrip("\r\n")
        batches = []
        selected = []

        async def respond(messages, info):
            system_prompt = next(
                part.content for message in messages for part in message.parts
                if isinstance(part, SystemPromptPart)
            )
            tool = info.output_tools[0]
            if "Categories" in tool.parameters_json_schema["properties"]:
                self.assertTrue(system_prompt.startswith(llm.read_md(llm.sys_prompt_categorie)))
                args = {"Categories": [category]}
            else:
                self.assertTrue(system_prompt.startswith(llm.read_md(llm.sys_prompt_konto)))
                _, batch = system_prompt.split(f"\n{description}\n", 1)
                batches.append(batch)
                rows = [json.loads(line) for line in batch.splitlines()]
                selected.append(rows[0])
                args = {"Kontos": [rows[0]]}
            return ModelResponse(parts=[ToolCallPart(tool.name, args)])

        self.model.return_value = FunctionModel(respond)
        with asyncio.Runner() as runner:
            runner.get_loop()
            result = llm.ask_model(f"Rechnung {description}", [])

        self.assertEqual("".join(batches), source)
        self.assertTrue(all(0 < len(batch) <= 30000 for batch in batches))
        self.assertEqual(result.splitlines(), [f'"Konto":"{row["Konto"]}"' for row in selected])
        expected_calls = 1 + len(batches)  # One category query, then account queries.
        self.assertEqual(self.server.call_count, expected_calls)
        self.assertEqual(self.ready.call_count, expected_calls)
        self.assertEqual(self.stop.call_count, expected_calls)
        self.assertEqual(self.model.call_count, expected_calls)
        self.sleep.assert_not_called()
        return len(batches)

    def test_full_pipeline_with_one_account_batch(self):
        self.assertEqual(self.run_pipeline("Eigenkapital"), 1)

    def test_full_pipeline_with_multiple_account_batches(self):
        self.assertGreater(self.run_pipeline("Sonstige_betriebliche_Aufwendungen"), 1)


class CallbackTests(SafeTestCase):
    def test_start_returns_greeting_for_chatbot_and_state(self):
        chatbot, state = llm.start()
        self.assertEqual(chatbot, state)
        self.assertEqual(len(chatbot), 1)
        self.assertEqual(chatbot[0]["role"], "assistant")
        self.assertIn("Rechnung", chatbot[0]["content"])
        self.assertIn("Hotelübernachtung", chatbot[0]["content"])

    def test_start_creates_fresh_messages_for_each_session(self):
        first, _ = llm.start()
        first[0]["content"] = "changed"
        second, _ = llm.start()
        self.assertNotEqual(first, second)

    def test_ask_model_passes_history_and_configured_prompts(self):
        history = [{"role": "user", "content": "previous"}]
        with patch.object(llm, "main", return_value="answer") as workflow:
            self.assertEqual(llm.ask_model("Rechnung Hotel", history), "answer")
        workflow.assert_called_once_with(
            "Rechnung Hotel", history, [llm.sys_prompt_categorie, llm.sys_prompt_konto, None], "local"
        )

    def test_first_start_passes_supplied_prompts(self):
        prompts = ["category", "konto"]
        with patch.object(llm, "main", return_value="answer") as workflow:
            self.assertEqual(llm.first_start("receipt", [], prompts), "answer")
        workflow.assert_called_once_with("receipt", [], prompts, "local")

    def test_callback_errors_are_not_swallowed(self):
        with patch.object(llm, "main", side_effect=RuntimeError("model failed")):
            for callback in (llm.ask_model, llm.first_start):
                with self.subTest(callback=callback.__name__), self.assertRaises(RuntimeError):
                    callback("receipt", [])


class GradioTests(SafeTestCase):
    def test_main_wires_callback_and_initial_state(self):
        with patch.object(app.gr, "Blocks") as blocks, patch.object(app.gr, "ChatInterface") as chat:
            app.main()
        blocks.assert_called_once_with(fill_height=True)
        chat.assert_called_once_with(fn=llm.ask_model)
        demo = blocks.return_value.__enter__.return_value
        demo.load.assert_called_once_with(
            llm.start, outputs=[chat.return_value.chatbot, chat.return_value.chatbot_state]
        )
        demo.launch.assert_called_once_with(inbrowser=True)

    def test_both_interface_helpers_launch_supplied_callback(self):
        callback = Mock()
        for module in (app, interface):
            with self.subTest(module=module.__name__), patch.object(module.gr, "ChatInterface") as chat:
                module.interface_run(callback)
                chat.assert_called_once_with(fn=callback)
                chat.return_value.launch.assert_called_once_with(inbrowser=True)

    def test_real_gradio_components_build_and_register_load_callback(self):
        self.launch.side_effect = None

        async def build_interface():
            # Build inside a running loop so Gradio does not allocate orphan
            # event loops for its locks and stop events.
            app.main()

        asyncio.run(build_interface())
        self.launch.assert_called_once()
        demo = self.launch.call_args.args[0]
        self.assertIsInstance(demo, app.gr.Blocks)
        self.assertTrue(any(fn.fn is llm.start for fn in demo.fns.values()))
        config = demo.get_config_file()
        self.assertTrue(any(component["type"] == "chatbot" for component in config["components"]))

    def test_script_entrypoint_builds_and_launches_ui(self):
        with patch.object(app.gr, "Blocks") as blocks, patch.object(app.gr, "ChatInterface") as chat:
            runpy.run_path(str(PROJECT_ROOT / "main.py"), run_name="__main__")
        chat.assert_called_once_with(fn=llm.ask_model)
        blocks.return_value.__enter__.return_value.launch.assert_called_once_with(inbrowser=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
