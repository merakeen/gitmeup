import argparse
import os
import re
import shlex
import subprocess
import sys
from textwrap import dedent
from pathlib import Path

from google import genai
from dotenv import dotenv_values

from gitmeup import __version__

# CONSTANT: Hard limit for diff size to prevent token exhaustion (429 errors).
# ~4 characters per token. 40,000 chars is roughly 10k tokens, leaving plenty of room.
MAX_DIFF_CHARS = 40000
# Long-lived, general-purpose text model for this CLI workflow.
# Live-audio and image-specialized models are intentionally not defaulted here.
DEFAULT_MODEL = "gemini-2.5-flash-lite"
CONVENTIONAL_TYPES = (
    "feat",
    "fix",
    "chore",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "ci",
    "revert",
)
CONVENTIONAL_HEADER_RE = re.compile(
    rf"^(?P<type>{'|'.join(CONVENTIONAL_TYPES)})(\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<description>.+)"
)

SYSTEM_PROMPT = dedent(
    """
You are a Conventional Commits writer. You generate precise commit messages that follow Conventional Commits 1.0.0:

<type>[optional scope]: <description>

Valid types include: feat, fix, chore, docs, style, refactor, perf, test, ci, and revert.
Use "!" or a BREAKING CHANGE footer for breaking changes.
Avoid non-standard types.
Suggest splitting changes into multiple commits when appropriate, and reflect that by outputting multiple git commit commands.

You receive:
- A `git diff --stat` output
- A `git status` output
- A `git diff` output (note: binary files and large lockfiles are excluded)

RULES FOR DECIDING COMMITS:
- Keep each commit atomic and semantically focused (feature, refactor, docs, locales, tests, CI, assets, etc.).
- Never invent files; operate only on files that appear in the provided git status or diff.
- If staged vs unstaged is unclear, assume everything is unstaged and must be added.
- If the changes are heterogeneous, split them into multiple commits and multiple batches.
- Scope must reflect the changed area from file paths (module/component/docs/tests/etc), not the repository or package name.
- When a batch spans multiple top-level areas, prefer no scope and split into smaller commits when possible.

STRICT PATH QUOTING (MANDATORY):
You output git commands that the user will paste directly in a POSIX shell.

For every path in git add/rm/mv:
- Quote the path with double quotes only if it contains characters outside the safe set [A-Za-z0-9._/\\-].
- Always quote paths containing: space, tab, (, ), [, ], {, }, &, |, ;, *, ?, !, ~, $, `, ', ", <, >, #, %, or any non-ASCII character.
- Never quote safe paths unnecessarily.
- Do not invent or "fix" paths; use exactly the paths you see, correctly quoted.

COMMAND GROUPING AND ORDER:
- Group files into small, meaningful batches.
- For each batch:
  - First output one or more git add/rm/mv commands.
  - Immediately after those, output one git commit -m "type[optional scope]: description" for that batch.
- Do not include git push or any remote-related commands.

OUTPUT FORMAT (VERY IMPORTANT):
- Respond with one fenced code block with language "bash".
- Inside that block, output only executable commands, one per line.
- No prose or comments.
- You may separate batches with a single blank line between them.

STYLE OF COMMIT MESSAGES:
- Descriptions are detailed, imperative, and specific.
- Commit header must strictly follow: type(scope): description (scope optional).
- Avoid generic scopes such as the repository/package name (for this project: "gitmeup").
"""
)


def load_env() -> None:
    """
    Load configuration from env files, without committing secrets.

    Precedence:
    - Existing environment variables are kept.
    - ~/.gitmeup.env (global, for secrets)
    - ./.env in the current working directory (per-project overrides)
    - CLI --api-key and --model override everything.
    """
    # Keep shell-exported values highest priority.
    existing_keys = set(os.environ.keys())

    # File precedence: ~/.gitmeup.env < ./.env
    merged = {
        **dotenv_values(Path.home() / ".gitmeup.env"),
        **dotenv_values(Path.cwd() / ".env"),
    }

    # Apply only keys that weren't already set in the process environment.
    for key, value in merged.items():
        if value is None or key in existing_keys:
            continue
        os.environ[key] = value


def run_git(args, check=True):
    result = subprocess.run(
        ["git"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout


def ensure_repo():
    try:
        out = run_git(["rev-parse", "--is-inside-work-tree"], check=True).strip()
    except SystemExit:
        print("gitmeup: not inside a git repository.", file=sys.stderr)
        sys.exit(1)
    if out != "true":
        print("gitmeup: not inside a git repository.", file=sys.stderr)
        sys.exit(1)


def collect_context():
    # Use HEAD to capture both staged and unstaged changes in the diff
    diff_stat = run_git(["diff", "--stat", "HEAD"], check=False)
    status = run_git(["status", "--short"], check=False)

    # Exclude patterns that bloat tokens but provide low semantic value
    diff_args = [
        "diff",
        "HEAD",
        "--",
        ".",
        # Images / Binaries
        ":(exclude)*.png",
        ":(exclude)*.jpg",
        ":(exclude)*.jpeg",
        ":(exclude)*.gif",
        ":(exclude)*.svg",
        ":(exclude)*.webp",
        ":(exclude)*.ico",
        # Lockfiles (Lead to fast token exhaustion)
        ":(exclude)package-lock.json",
        ":(exclude)yarn.lock",
        ":(exclude)pnpm-lock.yaml",
        ":(exclude)bun.lockb",
        ":(exclude)poetry.lock",
        ":(exclude)Gemfile.lock",
        ":(exclude)go.sum",
        ":(exclude)Cargo.lock",
        ":(exclude)*.lock",
        # Minified / Generated code
        ":(exclude)*.min.js",
        ":(exclude)*.min.css",
        ":(exclude)*.map",
        ":(exclude)dist/*",
        ":(exclude)build/*",
        ":(exclude).next/*",
    ]
    diff = run_git(diff_args, check=False)
    return diff_stat, status, diff


def build_user_prompt(diff_stat, status, diff):
    # Truncate diff if it's still too massive
    if len(diff) > MAX_DIFF_CHARS:
        diff = (
            diff[:MAX_DIFF_CHARS]
            + "\n\n... [DIFF TRUNCATED BY GITMEUP TO SAVE TOKENS] ..."
        )

    parts = [
        "# git diff --stat",
        diff_stat.strip() or "(no diff stat)",
        "",
        "# git status --short",
        status.strip() or "(no status)",
        "",
        "# git diff (lockfiles & binaries excluded)",
        diff.strip() or "(no textual diff)",
        "",
        "# TASK",
        "Based on the changes above, propose git add/rm/mv and git commit commands as per the instructions.",
        "If the diff was truncated, rely on the file paths in the stat section to infer context.",
    ]
    return "\n".join(parts)


def call_llm(model, api_key, user_prompt):
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.0,
        },
    )
    return resp.text


def extract_bash_block(text):
    """Extract first ```bash ... ``` block. Return its inner content."""
    in_block = False
    lang_ok = False
    lines = []

    for line in text.splitlines():
        if line.startswith("```"):
            fence = line.strip()
            if not in_block:
                lang = fence[3:].strip()
                lang_ok = lang == "" or lang.lower() in {"bash", "sh", "shell"}
                in_block = True
                continue
            else:
                break
        elif in_block and lang_ok:
            lines.append(line)

    return "\n".join(lines).strip()


def parse_commands(block):
    commands = []
    buffered_lines = []
    start_line = None

    for line_number, raw in enumerate(block.splitlines(), start=1):
        # When we're not buffering, allow some light cleanup of common LLM artifacts.
        if not buffered_lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$ "):
                line = line[2:].lstrip()
        else:
            # Preserve raw line breaks while waiting for a closing quote.
            line = raw

        if not buffered_lines:
            buffered_lines = [line]
            start_line = line_number
        else:
            buffered_lines.append(line)

        candidate = "\n".join(buffered_lines)
        try:
            parsed = shlex.split(candidate)
        except ValueError as exc:
            if "No closing quotation" in str(exc):
                continue
            raise ValueError(
                f"Invalid shell syntax near line {start_line}: {exc}"
            ) from exc

        if parsed:
            commands.append(parsed)
        buffered_lines = []
        start_line = None

    if buffered_lines:
        snippet = " ".join(part.strip() for part in buffered_lines if part.strip())
        raise ValueError(
            f"Unterminated quoted string starting near line {start_line}: {snippet}"
        )

    if not commands:
        raise ValueError("No executable commands found in bash block.")

    return commands


def _parse_status_porcelain_z_paths(porcelain_z):
    """
    Parse `git status --porcelain -z` and return all paths present in entries.
    Includes both source and destination for renames/copies.
    """
    paths = set()
    entries = porcelain_z.split("\0")
    index = 0

    while index < len(entries):
        entry = entries[index]
        if not entry:
            index += 1
            continue
        if len(entry) < 4:
            index += 1
            continue

        xy = entry[:2]
        path = entry[3:]
        if path:
            paths.add(path)

        # In -z porcelain mode, rename/copy emits an extra NUL-delimited path.
        if "R" in xy or "C" in xy:
            index += 1
            if index < len(entries):
                other_path = entries[index]
                if other_path:
                    paths.add(other_path)

        index += 1

    return paths


def _build_casefold_path_index():
    """
    Build case-insensitive path index from tracked files + current status entries.
    """
    tracked = run_git(["ls-files", "-z"], check=False)
    status = run_git(["status", "--porcelain", "-z"], check=False)

    paths = {p for p in tracked.split("\0") if p}
    paths.update(_parse_status_porcelain_z_paths(status))

    index = {}
    for path in paths:
        key = path.casefold()
        index.setdefault(key, []).append(path)

    for key in index:
        index[key].sort()

    return index


def _resolve_path_casing(path, path_index):
    candidates = path_index.get(path.casefold())
    if not candidates:
        return path
    if path in candidates:
        return path
    if len(candidates) == 1:
        return candidates[0]

    options = ", ".join(shlex.quote(candidate) for candidate in candidates)
    raise ValueError(
        f"Ambiguous case-insensitive match for path {shlex.quote(path)}. "
        f"Candidates: {options}"
    )


def _path_indices_for_git_path_command(cmd):
    if len(cmd) < 3 or cmd[0] != "git" or cmd[1] not in {"add", "rm", "mv"}:
        return []

    if "--" in cmd:
        sep_index = cmd.index("--")
        return list(range(sep_index + 1, len(cmd)))

    # Fallback when `--` is omitted: treat non-option args as paths.
    return [
        index
        for index, arg in enumerate(cmd[2:], start=2)
        if arg and not arg.startswith("-")
    ]


def normalize_command_paths(commands):
    path_commands = [cmd for cmd in commands if _path_indices_for_git_path_command(cmd)]
    if not path_commands:
        return commands, []

    path_index = _build_casefold_path_index()
    normalized = []
    corrections = []

    for command_index, cmd in enumerate(commands, start=1):
        normalized_cmd = list(cmd)
        for path_index_in_cmd in _path_indices_for_git_path_command(normalized_cmd):
            original_path = normalized_cmd[path_index_in_cmd]
            resolved_path = _resolve_path_casing(original_path, path_index)
            if resolved_path != original_path:
                normalized_cmd[path_index_in_cmd] = resolved_path
                corrections.append((command_index, original_path, resolved_path))
        normalized.append(normalized_cmd)

    return normalized, corrections


def _extract_commit_message_headers(cmd):
    if len(cmd) < 2 or cmd[0] != "git" or cmd[1] != "commit":
        return []

    messages = []
    index = 2
    while index < len(cmd):
        arg = cmd[index]
        if arg in {"-m", "--message"}:
            if index + 1 >= len(cmd):
                raise ValueError("git commit command uses -m/--message without a value.")
            messages.append(cmd[index + 1])
            index += 2
            continue
        index += 1

    if not messages:
        raise ValueError("git commit command must include -m/--message.")

    headers = []
    for message in messages:
        first_line = message.splitlines()[0] if message else ""
        headers.append(first_line.strip())
    return headers


def _project_generic_scopes():
    scopes = {Path.cwd().name.casefold(), "gitmeup"}
    pyproject = Path.cwd() / "pyproject.toml"
    if not pyproject.exists():
        return scopes

    in_project_section = False
    try:
        for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project_section = line == "[project]"
                continue
            if not in_project_section or not line.startswith("name"):
                continue
            match = re.match(r'name\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                scopes.add(match.group(1).strip().casefold())
            break
    except OSError:
        # Validation can continue even if pyproject cannot be read.
        pass
    return scopes


def _is_commit_command(cmd):
    return len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "commit"


def _iter_commit_batches(commands):
    batch_paths = []
    for command_index, cmd in enumerate(commands, start=1):
        for path_index in _path_indices_for_git_path_command(cmd):
            batch_paths.append(cmd[path_index])

        if _is_commit_command(cmd):
            yield command_index, cmd, list(batch_paths)
            batch_paths = []


def _batch_top_level_areas(paths):
    areas = set()
    for path in paths:
        normalized = path[2:] if path.startswith("./") else path
        if not normalized or normalized == ".":
            continue
        areas.add(normalized.split("/", 1)[0].casefold())
    return sorted(areas)


def validate_commit_messages(commands):
    generic_scopes = _project_generic_scopes()
    for command_index, cmd, batch_paths in _iter_commit_batches(commands):
        headers = _extract_commit_message_headers(cmd)
        if not headers:
            continue

        header = headers[0]
        if not header:
            raise ValueError(f"Command {command_index}: commit header is empty.")

        header_match = CONVENTIONAL_HEADER_RE.match(header)
        if not header_match:
            raise ValueError(
                f"Command {command_index}: invalid Conventional Commit header {header!r}. "
                "Expected: <type>(scope): <description> (scope optional)."
            )

        scope = header_match.group("scope")
        if not scope:
            continue

        scope_key = scope.strip().casefold()
        if scope_key in generic_scopes:
            raise ValueError(
                f"Command {command_index}: scope {scope!r} is too generic. "
                "Use a path-derived area scope or omit scope."
            )

        areas = _batch_top_level_areas(batch_paths)
        if len(areas) > 1:
            raise ValueError(
                f"Command {command_index}: scoped commit spans multiple top-level areas "
                f"({', '.join(areas)}). Split the batch or omit scope."
            )


def run_commands(commands, apply):
    try:
        commands, corrections = normalize_command_paths(commands)
        validate_commit_messages(commands)
    except ValueError as exc:
        print(f"gitmeup: {exc}", file=sys.stderr)
        sys.exit(1)

    if corrections:
        print("Adjusted path casing to match repository paths:\n")
        for command_index, original_path, resolved_path in corrections:
            print(
                f"- command {command_index}: "
                f"{shlex.quote(original_path)} -> {shlex.quote(resolved_path)}"
            )
        print()

    print("Proposed commands:\n")
    for cmd in commands:
        print(" ".join(shlex.quote(part) for part in cmd))

    if not apply:
        print("\nDry run: not executing commands. Re-run with --apply to execute.")
        return

    print("\nExecuting commands...\n")
    for cmd in commands:
        print("+", " ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Command failed with exit code {result.returncode}. Aborting.",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    print("\nCommands executed.\n")


def main(argv=None):
    # Load env from ~/.gitmeup.env and ./ .env before reading os.environ
    load_env()

    parser = argparse.ArgumentParser(
        prog="gitmeup",
        description="Generate Conventional Commits from current git changes using Gemini.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GITMEUP_MODEL", DEFAULT_MODEL),
        help=f"Gemini model name (default: {DEFAULT_MODEL} or $GITMEUP_MODEL).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute generated git commands. Without this flag, just print them.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key (default: $GEMINI_API_KEY).",
    )
    parser.add_argument(
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show gitmeup version and exit.",
    )

    args = parser.parse_args(argv)

    if not args.api_key:
        print(
            "Missing Gemini API key. Set GEMINI_API_KEY or use --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    ensure_repo()

    porcelain = run_git(["status", "--porcelain"], check=False)
    if porcelain.strip() == "":
        print("Working tree clean. Nothing to commit.")
        sys.exit(0)

    diff_stat, status, diff = collect_context()
    prompt = build_user_prompt(diff_stat, status, diff)

    # Calculate rough token usage for user awareness (optional, but helpful for debugging)
    # print(f"DEBUG: Prompt size is approx {len(prompt)} characters.")

    raw_output = call_llm(args.model, args.api_key, prompt)

    bash_block = extract_bash_block(raw_output)

    if not bash_block:
        print(
            "gitmeup: failed to extract bash command block from model output.",
            file=sys.stderr,
        )
        print("Raw output:\n", raw_output)
        sys.exit(1)

    try:
        commands = parse_commands(bash_block)
    except ValueError as exc:
        print(f"gitmeup: failed to parse bash commands: {exc}", file=sys.stderr)
        print("Model output block:\n", file=sys.stderr)
        print(bash_block, file=sys.stderr)
        sys.exit(1)

    run_commands(commands, apply=args.apply)

    print("\nFinal git status:\n")
    print(run_git(["status", "-sb"], check=False))

    print("Review your history with:")
    print("  git log --oneline --graph --decorate -n 10")


if __name__ == "__main__":
    main()
