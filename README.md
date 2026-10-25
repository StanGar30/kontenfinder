# Kontenfinder AI Agent

A local assistant for finding German bookkeeping accounts from a receipt description. Describe what was bought and why, and the app suggests account numbers from the included **SKR04** chart of accounts.

The browser interface is built with Gradio. Under the hood, Pydantic AI sends structured requests to a local Bonsai model through llama.cpp. No cloud model API key is required.

[Installation](#installation) · [Usage](#usage) · [How it works](#how-it-works) · [Troubleshooting](#troubleshooting)

## System requirements

Recommended for Bonsai 2 27B (`PQ2_0`) with the app's 32,768-token context:

| Component | Requirement / recommendation |
| --- | --- |
| OS | Windows x64 with the supplied launcher. Linux and macOS require the code changes described in the model setup section. |
| RAM | 32 GB recommended. |
| GPU / VRAM | NVIDIA GPU compatible with the selected CUDA build; 16 GB VRAM recommended. |
| CUDA / driver | An NVIDIA driver supporting the installed CUDA runtime, plus the runtime DLLs shipped with the binaries. The [Bonsai setup](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/setup.ps1) selects CUDA 12.4 or 13.3 according to driver support. |

These memory figures are planning recommendations, not measured minimums. The [model weights alone occupy about 7.2 GB](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#memory-requirement); context and runtime buffers need additional memory. Actual usage depends on the runtime, GPU offload, and other running applications.

## Installation

### 1. Get the project

Clone this repository or download and extract it. Open a terminal in the project folder, where `main.py` and `requirements.txt` are located.

You will need Python 3.11 or 3.12, a browser, and a machine capable of running the local model. Internet access is needed to download dependencies and model files.

### 2. Install the Python dependencies

Create a virtual environment:

```sh
python -m venv .venv
```

Activate it using the command for your shell:

| Shell | Command |
| --- | --- |
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| bash / zsh | `source .venv/bin/activate` |

Then install the dependencies:

```sh
python -m pip install -r requirements.txt
```

### 3. Set up the local model

**Model weights and llama.cpp binaries are not included in this repository. Download and set them up separately before running the app.**

The current configuration uses **Bonsai 2 27B** in the `PQ2_0` GGUF format:

- Download the model from [PrismML on Hugging Face](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
- Use the setup instructions in [Bonsai Demo](https://github.com/PrismML-Eng/Bonsai-demo) to install the matching PrismML llama.cpp binaries. The `PQ2_0` model requires that fork.

Keep the runtime inside a `Bonsai-demo` folder in the project root. The current launcher expects this layout:

```text
lexware_ai_agent/
├── main.py
├── llm_req.py
└── Bonsai-demo/
    ├── bin/
    │   └── cuda/
    │       ├── llama-server.exe
    │       └── ...                  # DLLs supplied with the runtime
    └── models/
        └── Ternary-Bonsai-2-27B-PQ2_0.gguf
```

If the Bonsai installer puts the weights in `models/bonsai2-gguf/27B/`, move the `*-PQ2_0.gguf` file directly into `Bonsai-demo/models/`. Keep only the intended model matching that pattern there: the launcher selects the first match.

The model path, server executable, and inference settings are defined in `run_llama_server()` in [llm_req.py](llm_req.py). The supplied app uses Windows and NVIDIA/CUDA. Using another platform also requires adapting process creation and shutdown in `llm_req.py` and `main.py`, which currently use Windows-specific APIs and `taskkill`. The shell commands above only set up the Python environment.

### 4. Start the app

With the virtual environment active, run:

```sh
python main.py
```

The chat opens in your browser. If it does not open automatically, use the local URL printed in the terminal.

On Windows, you can also double-click `Run.bat`. It uses the project's `.venv` and writes console output to `log.txt`. `Run.vbs` starts the same launcher with the console hidden.

## Usage

Start each message with **`Rechnung`**, followed by a description of the receipt. The keyword is case-insensitive; a colon is optional.

For example:

```text
Rechnung: Druckerpapier, Kugelschreiber und Ordner für das Büro.
```

```text
Rechnung: Monatliche Miete für ausschließlich betrieblich genutzte Büroräume.
```

```text
Rechnung: Ein Einzelhändler kauft Waren zum unveränderten Weiterverkauf.
```

Include the purpose of the purchase and whether it is for business or private use. For purchases involving the owner or a shareholder, the company type can also help distinguish the relevant accounts.

The reply contains suggested account numbers, one per line. For example, the entry for office supplies is displayed as:

```text
"Konto":"06815"
```

Use the suggestions when choosing an account for your booking in Lexware. The app works with typed descriptions and displays account numbers; account names and other details are available in the included `Kontenrahmen_categories_*.jsonl` files.

> AI suggestions do not guarantee correct accounting classification. Always verify the proposed accounts and the final choice before recording a transaction.

Each message is processed independently, so include the relevant details in every request. Processing can take a while: the app loads the model for each classification stage and may split larger account lists into several requests.

Click **Exit the program** to shut down the app and its child processes. Closing the browser tab also triggers shutdown.

## How it works

1. The model selects relevant accounting categories from the receipt description.
2. The app loads the accounts belonging to those categories from the JSONL files.
3. A second model stage selects candidate accounts, and the app displays their numbers.

Requests use Pydantic schemas for structured responses. Model inference runs locally through `llama-server` at `http://127.0.0.1:8080`.

| File | Purpose |
| --- | --- |
| `main.py` | Browser interface and application shutdown |
| `llm_req.py` | Model lifecycle, classification, and response formatting |
| `prompt_get_categorie_en.md` | Instructions for category selection |
| `prompt_get_konto_en.md` | Instructions for account selection |
| `Kontenrahmen_categories_*.jsonl` | Account catalogue grouped by category |
| `requirements.txt` | Python dependencies |
| `Run.bat` / `Run.vbs` | Windows launchers |

The app loads the two `_en.md` prompts. Keep them and the JSONL files alongside `llm_req.py` when moving or distributing the project.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `Model *-PQ2_0.gguf not found` | Put the model directly in `Bonsai-demo/models/`. The launcher does not search subfolders. |
| The server executable cannot be found | Check `Bonsai-demo/bin/cuda/llama-server.exe` and its accompanying DLLs, or update the executable path in `run_llama_server()`. |
| The server exits or runs out of GPU memory | Read the terminal output or `log.txt`. Check model compatibility and available memory; GPU offload and context size are configured in `run_llama_server()`. |
| The model times out while loading | The startup timeout is 180 seconds. Check the server output before increasing it in `wait_for_server()`. |
| The app asks for a receipt description | Begin the message with `Rechnung` and add a description containing words. |
| The app keeps retrying | Inspect the logged `Req Error` messages. Model request failures currently retry until a response succeeds; exit and restart after resolving the cause. |

Keep port `8080` free before submitting a request. The app starts and stops its own model server.
