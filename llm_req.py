from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import capture_run_messages
from pydantic_ai.usage import UsageLimits
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Union
import subprocess
import json
import time
import re
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

BASE_DIR = Path(__file__).resolve().parent

sys_prompt_categorie = BASE_DIR / "prompt_get_categorie_en.md"
sys_prompt_konto = BASE_DIR / "prompt_get_konto_en.md"
categories_dir = BASE_DIR
workspace_home = BASE_DIR

category_files = {
    "Anlagevermögen":
        categories_dir / "Kontenrahmen_categories_Anlagevermögen.jsonl",

    "Umlaufvermögen":
        categories_dir / "Kontenrahmen_categories_Umlaufvermögen.jsonl",

    "Eigenkapital":
        categories_dir / "Kontenrahmen_categories_Eigenkapital.jsonl",

    "Fremdkapital":
        categories_dir / "Kontenrahmen_categories_Fremdkapital.jsonl",

    "Betriebliche_Erträge":
        categories_dir / "Kontenrahmen_categories_Betriebliche_Erträge.jsonl",

    "Materialaufwand":
        categories_dir / "Kontenrahmen_categories_Materialaufwand.jsonl",

    "Sonstige_betriebliche_Aufwendungen":
        categories_dir / "Kontenrahmen_categories_Sonstige_betriebliche_Aufwendungen.jsonl",

    "Weitere_Erträge_und_Aufwendungen":
        categories_dir / "Kontenrahmen_categories_Weitere_Erträge_und_Aufwendungen.jsonl",

    "Nicht_belegte_Kontenklasse":
        categories_dir / "Kontenrahmen_categories_Nicht_belegte_Kontenklasse.jsonl",

    "Vortrags_und_statistische_Konten":
        categories_dir / "Kontenrahmen_categories_Vortrags_und_statistische_Konten.jsonl",

    "Debitoren":
        categories_dir / "Kontenrahmen_categories_Debitoren.jsonl",
}

class UserProfile_Category(BaseModel):
    Categories: list[str] = Field(description="Category keys in descending order of relevance") 
    
class Konto(BaseModel):
    Konto: str
    data: list[str]

class UserProfile_Kontos(BaseModel):
    Kontos: list[Konto]
    
class Empty_dict(BaseModel):
    Answer: list[str]

# Wait until the local server is ready or the timeout expires.
def wait_for_server(process: subprocess.Popen, timeout: float = 180):
    deadline = time.monotonic() + timeout

    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("llama-server timed out while loading the model")

        try:
            with urlopen(
                "http://127.0.0.1:8080/health",
                timeout=min(2.0, remaining),
            ) as response:
                if response.status == 200:
                    return

        except HTTPError as error:
            if error.code != 503:
                raise
            # The server is responding, but the model is still loading.

        except (URLError, TimeoutError):
            # The server is not accepting connections yet or did not respond in time.
            pass

        time.sleep(0.25)

# Start the local llama server with the configured model.
def run_llama_server():
    base_dir = BASE_DIR / "Bonsai-demo"
    model = next(
        (base_dir / "models").glob("*-PQ2_0.gguf"),
        None,
    )
    if model is None:
        raise FileNotFoundError("Model *-PQ2_0.gguf not found")
    
    process = subprocess.Popen(
         [
            str(base_dir / "bin/cuda/llama-server.exe"),
            "-m", str(model),
            "--no-mmproj", "-ngl", "99", "-fa", "on",
            "-c", "32768",
            "--temp", "1.0", "--top-p", "0.95",
            "--top-k", "20", "--min-p", "0.05",
            "--jinja", "--alias", "bonsai2",
            "-np", "1", "-t", "8", "-tb", "8", "-ub", "256",
        ],
        cwd=base_dir,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    print(f"Runned llama-server PID: {process.pid}")

    return process

# Stop the server and its child processes.
def stop_server_tree(process: subprocess.Popen):
    """Reliably stop the process and all its child processes on Windows."""
    if process.poll() is not None:
        print("Process Finished.")
        return
    
    pid = process.pid
    print(f"Process Tree breaked for PID {pid}...")
    
    # taskkill /T terminates the process tree; /F forces termination.
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False  # Do not raise an exception if the process has already exited.
    )
    
    # Wait for the parent process to finish.
    try:
        process.wait(timeout=5)
        print("Process Tree breaked.")
    except subprocess.TimeoutExpired:
        print("Process Tree not breaked taskkill.")

# Run a structured request against the local model and stop the server afterward.
def local_model_request(
    message: str = None, 
    sys_prompt: Any = None, 
    profile: UserProfile_Category | UserProfile_Kontos | Empty_dict | None = None,
    kontos: str | None = None
    ) -> Union[dict, str]:
    proc = run_llama_server()
    try:
        wait_for_server(proc)
        if type(sys_prompt) is not str:
            sys_prompt_str = read_md(sys_prompt)
        else:
            sys_prompt_str = sys_prompt
        
        model = OpenAIChatModel(
            'local-model',
            provider=OpenAIProvider(
                base_url='http://127.0.0.1:8080/v1',
                api_key='dummy',
            ),
        )

        agent = Agent(
            model,
            output_type=profile,
            system_prompt=sys_prompt_str + "\n" + message + "\n" + kontos,
        )

        while True:
            try:
                with capture_run_messages() as messages:
                        start_time = time.perf_counter()
                        result = agent.run_sync(
                            message,
                            usage_limits=UsageLimits(request_limit=3),
                        )
                        end_time = time.perf_counter()
                        print("GenerationTime:", end_time - start_time)

                        json_string = result.output.model_dump_json()
                        break  # The request and serialization completed successfully.

            except Exception as error:
                print(f"Req Error: {error}. Rerun 2 sec.")
                time.sleep(2)
        # If raw JSON text is needed:
        json_string = result.output.model_dump_json()
        return json_string
    finally:
        stop_server_tree(proc)

# Read the system prompt from a UTF-8 Markdown file.
def read_md(sys_prompt: Any = None) -> str:
    with open(sys_prompt, "r", encoding="utf-8") as f:
        sys_prompt_str = f.read()
    return sys_prompt_str

# Request relevant categories and validate their keys.
def step_one_get_category(message: str = None, sys_prompt_list: str = None) -> str:
    for attempt in range(2):
        try:
            category_answer = local_model_request(
                message=message,
                sys_prompt=sys_prompt_list,
                profile=UserProfile_Category,
                kontos=""
            )
            categories = json.loads(category_answer)
            for category in categories["Categories"]:
                if category not in category_files:
                    raise ValueError(
                        f"Unknown category: {category}"
                    )
            break

        except Exception as e:
            print(f"Req Error {attempt + 1}: {e}")

            if attempt == 1:
                raise

            time.sleep(2)

    print("Answer for first req:", category_answer)
    
    return category_answer

# Request matching accounts and format their account numbers.
def step_two_get_konto(
    message: str = None, 
    sys_prompt_list: str = None, 
    category: Union[dict, str] = None, 
    all_kontos: str | None = None):
    for attempt in range(2):
        try:
            # Step 2. Automatically send a new request.
            # The receipt description stays the same, but the prompt is different.
            konto_answer = local_model_request(
                message=message,
                sys_prompt=sys_prompt_list,
                profile=UserProfile_Kontos,
                kontos=all_kontos
            )
            break
            
        except Exception as e:
            print(f"Req Error {attempt + 1}: {e}")

            if attempt == 1:
                raise

            time.sleep(2)
    konto_answer_json = json.loads(konto_answer)
    return "\n".join(
        f'"Konto":"{konto["Konto"]}"'
        for konto in konto_answer_json["Kontos"]
    )

# Combine account data from the selected category files.
def get_all_kontos(categories_list: list[str] | str | None = None) -> list[str]:
    all_kontos = ""
    categories = json.loads(categories_list)
    for category in categories["Categories"]:
        category_file = category_files[category]
        with open(category_file, "r", encoding='utf-8') as f:
            content = f.read().rstrip("\r\n")
            if content:
                all_kontos += ("\n" if all_kontos else "") + content
    return all_kontos

# Validate the receipt description and find matching accounts in batches.
def main(message: str = None, history: str = None, sys_prompt: list = None, model_type: str = "local") -> Union[dict, str]:
    if model_type != "local":
        raise ValueError(f"Unknown model: {model_type}")
    if model_type == "codex":
        print("Model not exist.")
    elif model_type == "local":
        if re.match(r"^\s*rechnung\b", message, flags=re.IGNORECASE):
            # Find all occurrences of "Rechnung" and "rechnung" in message.
            # Get their positions or remove them directly from message.
            # Check message for other words; skip if none remain, otherwise proceed.
            cleaned, count = re.subn(
                r"\brechnung\b",
                "",
                message,
                flags=re.IGNORECASE,
            )

            # The word Rechnung was present, and other words remain after removal.
            if count > 0 and re.search(r"[^\W\d_]", cleaned):
                message = cleaned.strip()
                category_answer = step_one_get_category(message, sys_prompt[0])
                all_kontos = get_all_kontos(category_answer)
                max_batch_chars = 30_000
                konto_lines = all_kontos.splitlines(keepends=True)

                batches = []
                current_batch = ""

                for line in konto_lines:
                    if current_batch and len(current_batch) + len(line) > max_batch_chars:
                        batches.append(current_batch)
                        current_batch = ""

                    current_batch += line

                if current_batch:
                    batches.append(current_batch)

                answer = ""

                for batch in batches:
                    answer += step_two_get_konto(
                        message,
                        sys_prompt[1],
                        category_answer,
                        batch
                    ) + "\n"
            else:
                answer = "Geben Sie 'Rechnung' mit einer vollständigen Beschreibung des Belegs ein. Das Wort 'Rechnung' muss unbedingt am Anfang stehen!"
        else:
            answer = "Geben Sie 'Rechnung' mit einer Beschreibung ein. Das Wort 'Rechnung' muss unbedingt am Anfang stehen!"
    return answer

# Handle a chat message using the configured local prompts.
def ask_model(message, history):
    answer, _ = start()
    answer = main(message, history, [sys_prompt_categorie, sys_prompt_konto, None], "local")

    return answer

# Process an initial message with the supplied prompts.
def first_start(message: str = None, history: str = None, first_prompt: Any = None) -> str:
    answer = main(message, history, first_prompt, "local")
    return answer

# Return the greeting for the chatbot and its state.
def start():
    """Display a greeting when the interface loads."""

    messages = [
        {
            "role": "assistant",
            "content": (
                "Geben Sie eine Beschreibung des Belegs ein, zum Beispiel: ""\n"
                "***Rechnung: Hotelübernachtung und Frühstück während einer Urlaubsreise.***""\n"
                "Das Wort 'Rechnung' muss am Anfang stehen."
            ),
        }
    ]
    return messages, messages

# result = ask_model("Rechnung: Schulhefte, Stifte, Rucksack und Bastelmaterial für das Kind.", "")
