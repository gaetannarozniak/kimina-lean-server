from anyio import BrokenResourceError, ClosedResourceError
from contextlib import asynccontextmanager
import asyncio
import logging
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

settings = Settings(max_repl_mem=32)
lean_directory = Path.home().resolve() / "model_generated_files"

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
    """Write a Lean file"""
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
        return f"Successfully wrote to Lean project: {file_path.absolute()}"
    except Exception as e:
        return f"Failed to write to project: {str(e)}"
    
@mcp.tool()
async def lean_check_file(absolute_file_path: str):
    "Return diagnostics for a given lean file, provided its absolute path"
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
    return await _run_lean_code(lean_code)

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

