r"""Integration tests against the REAL llama-server and loaded model.

Run against an already running server at the application's configured address::

    .venv\Scripts\python.exe -X utf8 tests\live_tests.py -v

Start the server with the application's launcher if it is not running::

    .venv\Scripts\python.exe -X utf8 tests\live_tests.py --start-server -v

Only a server started by this runner is stopped at the end. Inference is real:
there are no TestModel/FunctionModel responses. For application pipeline tests,
only process ownership is adapted so local_model_request reuses the running
server instead of starting and stopping a second one. These calls run in a
child Python process with a deadline because the application's retry loop is
unbounded. The application's model name, prompts, schema, and inference
settings are preserved in those tests.

The module name intentionally does not match test*.py: ordinary unittest
discovery continues to run the fast offline suite without loading the GPU.
Results and actual model responses are written next to this file.

Semantic cases are unambiguous engineering smoke checks derived from the
project's prompts and account catalogue, not an accountant-reviewed benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

import llm_req as llm


BASE_URL = "http://127.0.0.1:8080"
MODEL_NAME = "local-model"  # Exactly the name used by local_model_request.
OBSERVATIONS = []
CATEGORY_CASES = {
    "private_owner_expense": (
        "Ein Einzelunternehmer bezahlt seine private Urlaubsreise vom Geschäftskonto. "
        "Die Reise hat keinerlei betrieblichen Anlass.",
        "Eigenkapital",
    ),
    "resale_goods": (
        "Ein Einzelhändler kauft Handelswaren ausschließlich zum unveränderten Weiterverkauf.",
        "Materialaufwand",
    ),
    "office_rent": (
        "Monatliche Miete der ausschließlich betrieblich genutzten Büroräume, "
        "keine private Nutzung und kein Erwerb einer Immobilie.",
        "Sonstige_betriebliche_Aufwendungen",
    ),
    "long_term_machine": (
        "Kauf einer neuen Produktionsmaschine für 40000 Euro netto. "
        "Die Maschine wird im eigenen Betrieb mindestens fünf Jahre genutzt, "
        "sie ist nicht für den Weiterverkauf bestimmt.",
        "Anlagevermögen",
    ),
}


def all_accounts():
    return {
        row["Konto"]: row
        for path in llm.category_files.values()
        for line in path.read_text(encoding="utf-8").splitlines()
        for row in [json.loads(line)]
    }


def jsonl(rows):
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)


def http_json(path, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(BASE_URL + path, data=body, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read().decode("utf-8")
        result = json.loads(raw)
        status = response.status
    OBSERVATIONS.append({
        "endpoint": path, "status": status,
        "seconds": round(time.monotonic() - started, 3), "response": result,
    })
    return status, result


class LiveEnvironment:
    def __init__(self):
        self.start_server = False
        self.timeout = 180.0
        self.pipeline_timeout = 600.0
        self.temperature = None
        self.max_output_tokens = None
        self.process = None
        self.server_log = None
        self.observations_name = "live_observations.json"

    def open(self):
        try:
            status, body = http_json("/health", timeout=3)
        except (URLError, TimeoutError, OSError):
            status, body = None, None
        if status == 200:
            print(f"Using existing llama-server at {BASE_URL}", flush=True)
            return
        if status == 503:
            print("An existing server is loading; waiting without starting another process...", flush=True)
            llm.wait_for_server(ExistingServerProcess(), timeout=180)
            return
        if not self.start_server:
            raise RuntimeError(
                f"llama-server is not ready at {BASE_URL}: {status}, {body}. "
                "Start it first or run this file with --start-server."
            )
        if status is not None:
            raise RuntimeError(
                f"Port 8080 already responds with HTTP {status}; "
                "a second server will not be started on the occupied port."
            )

        self.server_log = (TEST_DIR / "llama_server.log").open("w", encoding="utf-8")
        original_popen = subprocess.Popen

        def launch_with_log(*args, **kwargs):
            kwargs.update(stdout=self.server_log, stderr=subprocess.STDOUT)
            return original_popen(*args, **kwargs)

        # This wrapper only redirects logs. The actual launcher, executable,
        # command-line arguments, model, and GPU configuration are unchanged.
        with patch.object(llm.subprocess, "Popen", side_effect=launch_with_log):
            self.process = llm.run_llama_server()
        print(f"Waiting for actual model load, PID {self.process.pid}...", flush=True)
        llm.wait_for_server(self.process, timeout=180)
        print("llama-server is ready", flush=True)

    def close(self):
        try:
            if self.process is not None:
                llm.stop_server_tree(self.process)
        finally:
            if self.server_log is not None:
                self.server_log.close()
            (TEST_DIR / self.observations_name).write_text(
                json.dumps(OBSERVATIONS, ensure_ascii=False, indent=2), encoding="utf-8"
            )


ENV = LiveEnvironment()


def setUpModule():
    unittest.addModuleCleanup(ENV.close)
    ENV.open()


class LiveCase(unittest.TestCase):
    def raw_chat(self, **overrides):
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply only with the word OK."}],
            "temperature": 0,
            "max_tokens": 128,
        }
        payload.update(overrides)
        status, body = http_json("/v1/chat/completions", payload, timeout=ENV.timeout)
        self.assertEqual(status, 200, json.dumps(body, ensure_ascii=False))
        return body

    def typed_request(self, profile, system_prompt, description, accounts=""):
        """Real OpenAI adapter + real HTTP + real Agent schema validation."""
        async def run():
            provider = OpenAIProvider(base_url=BASE_URL + "/v1", api_key="dummy")
            agent = Agent(
                OpenAIChatModel(MODEL_NAME, provider=provider),
                output_type=profile,
                system_prompt=system_prompt + "\n" + description + "\n" + accounts,
            )
            started = time.monotonic()
            messages = []
            settings = {}
            if ENV.temperature is not None:
                settings["temperature"] = ENV.temperature
            if ENV.max_output_tokens is not None:
                settings["max_tokens"] = ENV.max_output_tokens
            try:
                with capture_run_messages() as messages:
                    result = await agent.run(
                        description,
                        model_settings=settings or None,
                        usage_limits=UsageLimits(request_limit=3),
                        timeout=ENV.timeout,
                    )
                output = result.output
                OBSERVATIONS.append({
                    "test": self.id(), "description": description,
                    "profile": profile.__name__,
                    "settings": settings,
                    "seconds": round(time.monotonic() - started, 3),
                    "output": output.model_dump(),
                })
                return output
            except Exception as error:
                OBSERVATIONS.append({
                    "test": self.id(), "description": description,
                    "profile": profile.__name__, "error": repr(error),
                    "settings": settings,
                    "seconds": round(time.monotonic() - started, 3),
                    "messages": json.loads(ModelMessagesTypeAdapter.dump_json(messages)),
                })
                raise
            finally:
                await provider.client.close()

        return asyncio.run(run())

    def assert_categories(self, output):
        self.assertGreaterEqual(len(output.Categories), 1)
        self.assertLessEqual(len(output.Categories), 3)
        self.assertEqual(len(output.Categories), len(set(output.Categories)))
        self.assertTrue(set(output.Categories) <= set(llm.category_files))
        self.assertNotIn("Nicht_belegte_Kontenklasse", output.Categories)


class ServerContractTests(LiveCase):
    def test_health_reports_ready(self):
        status, body = http_json("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "ok")

    def test_openai_models_endpoint_lists_loaded_model(self):
        status, body = http_json("/v1/models")
        self.assertEqual(status, 200)
        self.assertTrue(body.get("data"), body)
        for model in body["data"]:
            self.assertIsInstance(model.get("id"), str)
            self.assertTrue(model["id"])

    def test_chat_completion_and_usage_match_openai_contract(self):
        body = self.raw_chat()
        self.assertEqual(body.get("object"), "chat.completion")
        self.assertTrue(body.get("id"))
        self.assertEqual(len(body["choices"]), 1)
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertTrue(choice["message"].get("content", "").strip(), body)
        self.assertEqual(choice["finish_reason"], "stop", body)
        usage = body["usage"]
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    def test_json_schema_response_format(self):
        body = self.raw_chat(
            messages=[{"role": "user", "content": 'Return exactly {"ready": true}.'}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "health_probe", "strict": True,
                    "schema": {
                        "type": "object", "properties": {"ready": {"type": "boolean"}},
                        "required": ["ready"], "additionalProperties": False,
                    },
                },
            },
        )
        self.assertEqual(json.loads(body["choices"][0]["message"]["content"]), {"ready": True})

    def test_streaming_returns_content_and_done_marker(self):
        payload = {
            "model": MODEL_NAME, "stream": True, "temperature": 0, "max_tokens": 128,
            "messages": [{"role": "user", "content": "Reply only with the word OK."}],
        }
        request = Request(
            BASE_URL + "/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        content, finish_reasons, done = [], [], False
        with urlopen(request, timeout=ENV.timeout) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                for choice in chunk.get("choices", []):
                    content.append(choice.get("delta", {}).get("content") or "")
                    if choice.get("finish_reason"):
                        finish_reasons.append(choice["finish_reason"])
        self.assertTrue(done, "Missing SSE [DONE] marker")
        self.assertTrue("".join(content).strip(), "No assistant content was streamed")
        self.assertEqual(finish_reasons, ["stop"])
        OBSERVATIONS.append({"test": self.id(), "streamed_content": "".join(content)})

    def test_malformed_request_returns_client_error_and_server_survives(self):
        status, body = http_json("/v1/chat/completions", {
            "model": MODEL_NAME, "messages": "messages must be an array",
        }, timeout=ENV.timeout)
        self.assertGreaterEqual(status, 400, body)
        self.assertLess(status, 500, body)
        self.assertIn("error", body)
        self.assertEqual(http_json("/health")[0], 200)


class StructuredOutputTests(LiveCase):
    def test_unicode_and_leading_zeros_survive_structured_output(self):
        row = {"Konto": "00100", "data": ["00100", "ÄÖÜ äöü ß €", "Описание чека", 'Text mit "Anführungen"']}
        output = self.typed_request(
            llm.UserProfile_Kontos,
            "Copy the supplied account object into Kontos exactly, including all data strings. "
            "Do not classify or modify the object.",
            "Copy the single supplied account.", jsonl([row]),
        )
        self.assertEqual(output.model_dump(), {"Kontos": [row]})

    def test_account_selection_preserves_full_catalogue_objects(self):
        catalogue = all_accounts()
        rows = [catalogue[number] for number in ("06310", "02100", "00100")]
        output = self.typed_request(
            llm.UserProfile_Kontos, llm.read_md(llm.sys_prompt_konto),
            CATEGORY_CASES["office_rent"][0], jsonl(rows),
        )
        self.assertTrue(output.Kontos)
        ids = [row.Konto for row in output.Kontos]
        self.assertIn("06310", ids)
        self.assertEqual(len(ids), len(set(ids)), "Duplicate account selections")
        allowed = {row["Konto"] for row in rows}
        for row in output.Kontos:
            with self.subTest(account=row.Konto):
                self.assertIn(row.Konto, allowed, "Model invented an account outside its input")
                self.assertEqual(row.model_dump(), catalogue[row.Konto], "Account data was rewritten or truncated")

    def test_empty_account_catalogue_does_not_produce_invented_accounts(self):
        output = self.typed_request(
            llm.UserProfile_Kontos, llm.read_md(llm.sys_prompt_konto),
            CATEGORY_CASES["office_rent"][0], "",
        )
        self.assertEqual(output.Kontos, [])

    def test_irrelevant_customer_accounts_return_no_matches(self):
        rows = [json.loads(line) for line in llm.category_files["Debitoren"].read_text(encoding="utf-8").splitlines()]
        output = self.typed_request(
            llm.UserProfile_Kontos, llm.read_md(llm.sys_prompt_konto),
            CATEGORY_CASES["office_rent"][0] + " Es handelt sich nicht um eine Kundenforderung.",
            jsonl(rows),
        )
        self.assertEqual(output.Kontos, [])


class CategoryQualityTests(LiveCase):
    def check_case(self, name):
        description, expected = CATEGORY_CASES[name]
        output = self.typed_request(
            llm.UserProfile_Category, llm.read_md(llm.sys_prompt_categorie), description,
        )
        self.assert_categories(output)
        self.assertEqual(output.Categories[0], expected, f"Case {name}: {output.Categories}")

    def test_private_owner_expense(self):
        self.check_case("private_owner_expense")

    def test_goods_bought_for_resale(self):
        self.check_case("resale_goods")

    def test_business_office_rent(self):
        self.check_case("office_rent")

    def test_long_term_production_machine(self):
        self.check_case("long_term_machine")


class ExistingServerProcess:
    """Non-owning process handle used only by the worker's lifecycle adapter."""

    returncode = None

    def poll(self):
        return None


def application_worker(payload):
    """Run actual app code; only launching/killing the shared server is replaced."""
    trace = []
    actual_request = llm.local_model_request

    def traced_request(*args, **kwargs):
        response = actual_request(*args, **kwargs)
        supplied = kwargs.get("kontos") or ""
        trace.append({
            "profile": kwargs["profile"].__name__,
            "input_accounts": [json.loads(line)["Konto"] for line in supplied.splitlines()],
            "input_characters": len(supplied), "output": json.loads(response),
        })
        return response

    with patch.object(llm, "run_llama_server", return_value=ExistingServerProcess()), \
         patch.object(llm, "stop_server_tree"), \
         patch.object(llm, "local_model_request", side_effect=traced_request):
        if payload["action"] == "pipeline":
            answer = llm.ask_model(payload["message"], [])
        else:
            profile = getattr(llm, payload["profile"])
            answer = llm.local_model_request(
                message=payload["message"], sys_prompt=payload["system_prompt"],
                profile=profile, kontos=payload.get("accounts", ""),
            )
    return {"answer": answer, "trace": trace}


class ApplicationIntegrationTests(LiveCase):
    def run_worker(self, payload):
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True, text=True, encoding="utf-8",
                timeout=ENV.pipeline_timeout, cwd=PROJECT_ROOT,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            OBSERVATIONS.append({
                "test": self.id(), "error": "application worker timeout",
                "seconds": round(time.monotonic() - started, 3), "stdout_tail": stdout[-6000:],
            })
            self.fail(f"Application exceeded {ENV.pipeline_timeout}s; possible endless retry.\n{stdout[-6000:]}")
        if result.returncode:
            OBSERVATIONS.append({
                "test": self.id(), "error": "application worker failed",
                "stdout_tail": result.stdout[-6000:], "stderr_tail": result.stderr[-6000:],
            })
        self.assertEqual(result.returncode, 0, result.stdout[-6000:] + result.stderr[-6000:])
        lines = [line for line in result.stdout.splitlines() if line.startswith("LIVE_RESULT:")]
        self.assertEqual(len(lines), 1, result.stdout[-6000:])
        response = json.loads(lines[0].removeprefix("LIVE_RESULT:"))
        OBSERVATIONS.append({
            "test": self.id(), "message": payload["message"],
            "seconds": round(time.monotonic() - started, 3), **response,
        })
        self.assertEqual(http_json("/health")[0], 200, "Worker stopped the shared server")
        return response

    def validate_pipeline(self, response, expected_category):
        trace = response["trace"]
        self.assertTrue(trace)
        category_calls = [item for item in trace if item["profile"] == "UserProfile_Category"]
        self.assertTrue(category_calls)
        # Earlier category responses can be rejected and retried by step one.
        categories = llm.UserProfile_Category.model_validate(category_calls[-1]["output"])
        self.assert_categories(categories)
        self.assertEqual(categories.Categories[0], expected_category)
        catalogue = all_accounts()
        selected = []
        account_calls = [item for item in trace if item["profile"] == "UserProfile_Kontos"]
        self.assertTrue(account_calls, "The account stage was never executed")
        for item in account_calls:
            self.assertGreater(item["input_characters"], 0)
            self.assertLessEqual(item["input_characters"], 30000)
            output = llm.UserProfile_Kontos.model_validate(item["output"])
            ids = [row.Konto for row in output.Kontos]
            self.assertEqual(len(ids), len(set(ids)), "Duplicate accounts within one response")
            for row in output.Kontos:
                with self.subTest(account=row.Konto):
                    self.assertIn(row.Konto, item["input_accounts"], "Selected account was not in this batch")
                    self.assertEqual(row.model_dump(), catalogue[row.Konto], "Account data differs from JSONL")
                    selected.append(row.Konto)
        displayed = [line for line in response["answer"].splitlines() if line.strip()]
        self.assertTrue(displayed, "The invoice produced no account suggestions")
        self.assertTrue(all(re.fullmatch(r'"Konto":"\d{5}"', line) for line in displayed))
        self.assertEqual(displayed, [f'"Konto":"{number}"' for number in selected])
        return selected, len(account_calls)

    def test_application_local_request_uses_real_server_and_schema(self):
        response = self.run_worker({
            "action": "request", "profile": "Empty_dict", "message": "Return OK.",
            "system_prompt": 'Return an object with Answer containing exactly one string: "OK".',
        })
        self.assertEqual(json.loads(response["answer"]), {"Answer": ["OK"]})

    def test_full_private_expense_pipeline(self):
        response = self.run_worker({
            "action": "pipeline", "message": "Rechnung " + CATEGORY_CASES["private_owner_expense"][0],
        })
        selected, _ = self.validate_pipeline(response, "Eigenkapital")
        self.assertTrue(set(selected) & {"02100", "02500"}, selected)

    def test_full_office_rent_pipeline_with_multiple_batches(self):
        response = self.run_worker({
            "action": "pipeline", "message": "RECHNUNG " + CATEGORY_CASES["office_rent"][0],
        })
        selected, batches = self.validate_pipeline(response, "Sonstige_betriebliche_Aufwendungen")
        self.assertGreater(batches, 1)
        self.assertIn("06310", selected)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-server", action="store_true", help="Start the project server if offline")
    parser.add_argument("--timeout", type=float, default=180, help="Deadline for individual API/Agent requests")
    parser.add_argument("--pipeline-timeout", type=float, default=600, help="Deadline for the app worker")
    parser.add_argument("--temperature", type=float, help="Optional component-test override; otherwise server default")
    parser.add_argument("--max-output-tokens", type=int, help="Optional component-test override; otherwise server default")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("tests", nargs="*", help="Optional ClassName.test_method selectors")
    args = parser.parse_args()
    if args.worker:
        result = application_worker(json.loads(sys.stdin.read()))
        print("LIVE_RESULT:" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if args.timeout <= 0 or args.pipeline_timeout <= 0:
        parser.error("Timeouts must be positive")
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    ENV.start_server = args.start_server
    ENV.timeout = args.timeout
    ENV.pipeline_timeout = args.pipeline_timeout
    ENV.temperature = args.temperature
    ENV.max_output_tokens = args.max_output_tokens
    classes = (ServerContractTests, StructuredOutputTests, CategoryQualityTests, ApplicationIntegrationTests)
    if args.tests:
        suite = unittest.defaultTestLoader.loadTestsFromNames(args.tests, module=sys.modules[__name__])
        report_name = "live_selected_results.txt"
        ENV.observations_name = "live_selected_observations.json"
    else:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
        report_name = "live_test_results.txt"
    with (TEST_DIR / report_name).open("w", encoding="utf-8") as report:
        stream = Tee(sys.stderr, report)
        stream.write(f"REAL llama-server integration tests: {BASE_URL}, model={MODEL_NAME}\n")
        stream.write("Generation is real. Process ownership alone is adapted for app worker calls.\n\n")
        stream.write(
            f"Component overrides: temperature={ENV.temperature}, max_tokens={ENV.max_output_tokens}; "
            "None means server default. App worker uses project defaults.\n\n"
        )
        with redirect_stdout(Tee(sys.stdout, report)):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
