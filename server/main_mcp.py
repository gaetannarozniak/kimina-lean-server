from anyio import BrokenResourceError, ClosedResourceError
from contextlib import asynccontextmanager
import asyncio
import logging
import re
from pathlib import Path
from uuid import uuid4
from kimina_client import ReplResponse, Snippet
from fastmcp import FastMCP
from fastapi import HTTPException
from starlette.requests import ClientDisconnect
import time
from typing import Annotated, Optional

from pydantic import Field

from server.manager import Manager
from server.routers.check import run_checks
from server.settings import Settings
from server.errors import ReplError, LeanError

logger = logging.getLogger("kimina-client")

settings = Settings(max_repl_mem=64, max_repls=64)
# lean_directory = Path.home().resolve() / "model_generated_files"
lean_directory = Path("/checkpoint/scientific-reasoning/gaetan/model_generated_files")

MATHLIB_DIR = (Path.cwd() / "mathlib4").resolve(strict = True)

manager = Manager(
    max_repls=settings.max_repls,
    max_repl_uses=settings.max_repl_uses,
    max_repl_mem=settings.max_repl_mem,
    init_repls=settings.init_repls,
)

async def _run_lean_code(lean_code: str):
    try:
        t = time.time()
        global manager
        snippet = Snippet(id=str(uuid4()), code=lean_code)
        results = await run_checks(
            snippets=[snippet],
            timeout=120.0,
            debug=False,
            manager=manager,
            reuse=True,
            infotree=None,
        )
        repl_response : ReplResponse = results[0]
        is_valid = repl_response.analyze().status == "valid"
        print(f"Time for running a Lean snippet: {time.time() - t}")
        return {
            "repl_response": repl_response,
            "is_valid": is_valid,
        }
    except ClientDisconnect:
        logger.warning("Client disconnected during request")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error="Client disconnected",
            ),
            "is_valid": False,
        }
    except asyncio.CancelledError:
        logger.warning("Task was cancelled (likely client disconnect)")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error="Task cancelled - client may have disconnected",
            ),
            "is_valid": False,
        }
    except TimeoutError:
        logger.error("REPL timeout in lean_run_code")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error="Lean REPL command timed out after 120 seconds",
                time=120.0,
            ),
            "is_valid": False,
        }
    except ReplError as e:
        logger.error(f"REPL error in lean_run_code: {e}")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error=f"REPL process error: {str(e)}. The REPL may have crashed or exceeded memory limits.",
            ),
            "is_valid": False,
        }
    except LeanError as e:
        logger.error(f"Lean error in lean_run_code: {e}")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error=f"Lean execution error: {str(e)}",
            ),
            "is_valid": False,
        }
    except HTTPException as e:
        logger.error(f"HTTP error in lean_run_code: {e.status_code} - {e.detail}")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error=f"Error: {e.detail}",
            ),
            "is_valid": False,
        }
    except (BrokenResourceError, ClosedResourceError) as e:
        logger.warning(f"Resource error in lean_run_code (client likely disconnected): {e}")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error="Connection closed - client may have disconnected",
            ),
            "is_valid": False,
        }
    except Exception as e:
        logger.exception(f"Unexpected error in lean_run_code: {e}")
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error=f"Unexpected error: {str(e)}",
            ),
            "is_valid": False,
        }

async def _event_loop_lag_monitor(interval: float = 0.2, warn_threshold: float = 0.05) -> None:
    """Periodically measures event loop lag. Logs a warning whenever the loop
    was blocked longer than warn_threshold seconds between two sleep wakeups."""
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(interval)
        lag = time.monotonic() - t0 - interval
        if lag > warn_threshold:
            logger.warning(f"[event-loop] blocked for {lag:.3f}s")

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    logger.info("Starting up: Importing Mathlib...")
    await _run_lean_code("import Mathlib")
    logger.info("Mathlib cached!")

    lag_monitor_task = asyncio.create_task(_event_loop_lag_monitor())

    yield

    lag_monitor_task.cancel()
    logger.info("Shutting down...")

mcp = FastMCP("Kimina MCP", lifespan=app_lifespan)

@mcp.tool()
async def lean_run_code(lean_code: str):
    """
    Run a Lean snippet provided as a string and return the Lean4 diagnostics.
    """
    return await _run_lean_code(lean_code)

@mcp.tool()
async def lean_write_file(
    code: Annotated[str, Field(description="Lean code, the content of the file")],
    file_name: Annotated[str, Field(description="An explicit name for the created file")],
    trajectory_id: str = "no_trajectory_id",
) -> str:
    """Write a Lean file and compile it"""
    global lean_directory

    if not file_name.endswith(".lean"):
        file_name += ".lean"

    lean_directory.mkdir(parents=True, exist_ok=True)
    file_path = lean_directory / trajectory_id / file_name

    if not "model_generated_files" in str(file_path):
        return f"Path to the file doesn't contain model_generated_files: {file_path}"
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code, encoding="utf-8")
    except Exception as e:
        return f"Failed to write to project: {str(e)}"

    with open(file_path, "r") as f:
        lean_code = f.read()
    compile_result = await _run_lean_code(lean_code)
    return f"Successfully wrote to Lean project: {file_path.absolute()}, result of the Lean evaluation: {compile_result}"
    
def _extract_candidate_lemma_names(lean_code: str) -> list[str]:
    """Extract candidate Mathlib lemma names explicitly written in Lean source code.

    Uses regex to find:
    - Qualified identifiers like Nat.add_comm, Finset.sum_comm (Namespace.name patterns)
    - Names inside tactic bracket lists: rw [X, ← Y], simp [X, Y], simp only [X]
    - Names in tactic application positions: exact X, apply X, have h := X, etc.

    False positives (keywords, types, etc.) are removed later by the Lean thmInfo check.
    """
    candidates: set[str] = set()

    # Qualified identifiers that start with an uppercase namespace component.
    # Matches: Nat.add_comm, List.map_id, Finset.sum_comm, Nat.Coprime.pow_dvd_of_pow_dvd …
    for m in re.finditer(r'\b([A-Z][A-Za-z0-9_]*(?:\.[A-Za-z_\'][A-Za-z0-9_\']*)+)\b', lean_code):
        candidates.add(m.group(1))

    # Names inside [...] bracket lists (rw, simp, simp only, …).
    # Also handles the ← reverse-rewrite prefix.
    for bracket_content in re.findall(r'\[([^\]]*)\]', lean_code):
        for item in bracket_content.split(','):
            item = item.strip().lstrip('←').strip()
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_\']*(?:\.[A-Za-z_\'][A-Za-z0-9_\']*)*)', item)
            if m and m.group(1):
                candidates.add(m.group(1))

    # Names in tactic application positions (catches unqualified names like sq_nonneg).
    # False positives (keywords like `by`, `fun`, types) are filtered by the thmInfo check.
    ident = r'([A-Za-z_][A-Za-z0-9_\']*(?:\.[A-Za-z_\'][A-Za-z0-9_\']*)*)'
    for pattern in [
        rf'\bexact\s+{ident}',
        rf'\bapply\s+{ident}',
        rf'\brefine\s+{ident}',
        rf'\buse\s+{ident}',
        rf':=\s+{ident}',       # have h : T := lemma_name ...
        rf'\bfrom\s+{ident}',
    ]:
        for m in re.finditer(pattern, lean_code):
            candidates.add(m.group(1))

    return list(candidates)


def _build_verification_snippet(candidates: list[str]) -> str:
    """Build a Lean #eval snippet that filters a list of name candidates to those
    that are theorem/lemma declarations (from Mathlib, Lean core, Std, etc.)."""
    if not candidates:
        return '\n#eval ("LEAN_LEMMAS_V1=" : String)\n'
    lean_list = ", ".join(f"`{name}" for name in candidates)
    return f"""
open Lean in
#eval show CoreM String from do
  let env ← getEnv
  let candidates : List Name := [{lean_list}]
  let lemmaNames := candidates.filter (fun n =>
    match env.find? n with
    | none => false
    | some (.thmInfo _) => true
    | _ => false)
  return "LEAN_LEMMAS_V1=" ++ String.intercalate "," (lemmaNames.map Name.toString)
"""


def _parse_lemmas_from_response(repl_response: ReplResponse) -> list[str]:
    """Extract Mathlib lemma names from a REPL response containing #eval output.

    Returns a sorted, deduplicated list of fully-qualified Lean declaration names.
    """
    if not repl_response.response:
        return []
    messages = repl_response.response.get("messages", [])
    lemmas: set[str] = set()
    for msg in messages:
        data = msg.get("data", "")
        m = re.search(r'LEAN_LEMMAS_V1=([^"]*)', data)
        if m:
            content = m.group(1).strip().strip(',')
            if content:
                lemmas.update(n for n in content.split(',') if n.strip())
    return sorted(lemmas)


@mcp.tool()
async def lean_check_file(
    absolute_file_path: str,
    return_lemmas_list: bool = False,
):
    """Return diagnostics for a given lean file, provided its absolute path.

    If return_lemmas_list is True and the proof is correct, also returns
    'mathlib_lemmas': a sorted list of fully-qualified Lean declaration names
    (e.g. "Nat.add_comm", "Finset.sum_comm") for every Mathlib lemma explicitly
    cited in the source code.  These names are globally unique and suitable for
    accumulating in a set to count distinct lemmas used.
    """
    global lean_directory

    def error_return(error_str: str):
        return {
            "repl_response": ReplResponse(
                id=str(uuid4()),
                error=error_str,
            ),
            "is_valid": False,
        }
    try:
        file_path: Path = Path(absolute_file_path).resolve(strict=True)
    except Exception as e:
        return error_return(str(e))

    if not lean_directory in file_path.parents:
        return error_return(f"The provided path {file_path} is not in the correct directory: {lean_directory}")
    with open(file_path, "r") as f:
        lean_code = f.read()

    result = await _run_lean_code(lean_code)

    if not result["is_valid"] or not return_lemmas_list:
        return result

    # Proof is valid — verify which explicitly cited names are Mathlib declarations.
    # This uses the already-cached import Mathlib REPL, so it is much cheaper than
    # re-running the proof.
    candidates = _extract_candidate_lemma_names(lean_code)
    verification_code = "import Mathlib\n\n" + _build_verification_snippet(candidates)
    verification_result = await _run_lean_code(verification_code)
    result["mathlib_lemmas"] = _parse_lemmas_from_response(verification_result["repl_response"])
    return result

@mcp.tool()
async def rg_in_mathlib(pattern: str, file_glob_pattern: Optional[str] = None):
    """
    Run ripgrep in the mathlib4 directory pattern with the provided pattern.
    Optionally restrict search to filenames matching file_pattern (glob).
    Return a maximum of 5 matches.

    Example
    -------
    Search for the regex "theorem.*Nat" only inside files under Algebra:

        await rg_in_mathlib(
            pattern="theorem.*Nat",
            file_glob_pattern="**/Algebra/**"
        )
    """

    max_matches = 5
    cmd = [
        "rg",
        "--no-config",
        "-n",
    ]
    if file_glob_pattern:
        cmd.extend(["--glob", file_glob_pattern])
    cmd.extend([
        "--", pattern,
        ".",
    ])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(MATHLIB_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert proc.stdout is not None

        lines = []
        while len(lines) < max_matches:
            line = await proc.stdout.readline()
            if not line:
                break
            lines.append(line.decode())

        proc.kill()
        await proc.wait()

        if not lines:
            return f"No matches found for pattern: {pattern}"

        return "".join(lines)

    except Exception as e:
        return f"Error running ripgrep: {str(e)}"

@mcp.tool()
async def read_mathlib_file(path_to_file: str, first_line: int, last_line: int):
    """
    Return the content of a file in Mathlib provided its relative path in the mathlib4 directory,
    between line number first_line and last_line (inclusive).
    Maximum 50 lines.
    Line numbers are 1-based.

    Example:
        read_mathlib_file(
            path_to_file="Mathlib/GroupTheory/OrderOfElement.lean",
            first_line=1,
            last_line=30,
        )
    """
    try:
        absolute_path = (MATHLIB_DIR / Path(path_to_file)).resolve(strict=True)
    except Exception as e:
        return {"error": str(e)}

    if first_line < 0 or last_line < first_line:
        return {"error": "Invalid line range"}

    last_line = min(last_line, first_line + 50)

    try:
        lines : list[str] = []
        with open(absolute_path, "r") as f:
            for i, line in enumerate(f, start=1):
                if i > last_line:
                    break
                if i >= first_line:
                    lines.append(line)

        if not lines:
            return {"error": "No content in requested range"}

        return {"content": "".join(lines)}

    except Exception as e:
        return {"error": str(e)} 


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")

