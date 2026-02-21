# gitmeup

`gitmeup` leverages popular LLMs to organize changes into atomic, semantic batches, and generates precise `git add` and `git commit` commands following the [Conventional Commits 1.0.0](https://www.conventionalcommits.org/) specification.



> [!IMPORTANT]
> It does **not** push anything, it just helps you decide *what* to commit and *how* to phrase it.
> It is designed for strict safety: it runs in **dry-run mode by default**, never pushes to remotes, and handles complex file paths with POSIX-compliant quoting.

### Navigation (only on Github)
* [What problem does it solve?](#what-problem-does-it-solve)
* [How does it work?](#how-it-works-in-practice)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Examples](#examples)
* [Behaviour](#behaviour)
* [License](#license)
* [Maintainer](#maintainer)

## What problem does it solve?

The typical workflow for complex changes involves significant manual overhead:

- Manually reviewing `git status` and `git diff` to determine logical split points.
- Crafting specific `git add` commands, risking missing files or including unrelated changes.
- Spending time formatting commit messages to adhere to Conventional Commits standards.
- Handling files with spaces, brackets, or special characters that require careful shell escaping.

`gitmeup` automates the staging and committing process:

- Groups changes into **atomic, semantically focused commits** (e.g., separating `feat` from `refactor` or `docs`).
- Generates precise `git add` sequences followed by `git commit -m "type(scope): description"`.
- Enforces **strict path quoting** to prevent shell expansion errors with complex filenames.
- Operates safely via a default **dry-run** mode, requiring explicit confirmation to execute.

## How it works (in practice)

From inside a git repository, `gitmeup` aggregates the following context:

- `git diff --stat HEAD`
- `git status --short`
- `git diff HEAD` with high-noise/low-value files excluded from the context body:
  - Images: `*.png`, `*.jpg`, `*.svg`, etc.
  - Lockfiles: `package-lock.json`, `yarn.lock`, `Cargo.lock`, etc. (to prevent token exhaustion).
  - Minified assets and map files.

This sanitized context is transmitted to an LLM model _(default: Gemini)_, which returns a single code block containing:

- Batches of `git add`, `git rm`, or `git mv` commands.
- Corresponding `git commit -m "..."` commands following the Conventional Commits specification.

You can then:

- Inspect the proposed command plan (default behavior).
- Execute the plan using `--apply`.

No `git push` or remote operations are ever generated.

## Installation

### From PyPI (recommended)

```bash
pip install gitmeup
```

This installs the `gitmeup` CLI into your environment.

### From source (editable dev install)

```bash
git clone [https://github.com/merakeen/gitmeup](https://github.com/merakeen/gitmeup)
cd gitmeup

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

## Configuration

`gitmeup` interacts with Google Gemini via `google-genai`. It needs:

* A Gemini API key
* A model name (default is `gemini-2.5-flash-lite` unless overridden)

Default rationale: this tool is text-first (`generate_content`), so we use a durable general Gemini 2.5 model instead of live-audio or image-specialized variants.

### 1. Secrets via env file (recommended)

`gitmeup` will automatically load:

1. `~/.gitmeup.env` (global, for secrets and defaults)
2. `./.env` in the current repo (for local overrides, optional)

Values in the environment override file values, and CLI flags override both.

Example global config:

```env
# ~/.gitmeup.env
GEMINI_API_KEY=your-gemini-key-here
GITMEUP_MODEL=gemini-2.5-flash-lite
```

> Keep `~/.gitmeup.env` out of any git repo. It lives only in your home directory.

Optional per-repo overrides:

```env
# ./.env (inside a project, usually without secrets if repo is shared)
GITMEUP_MODEL=gemini-2.5-pro
```

If you use a local `.env` with secrets, **ensure** `.env` is listed in that repo’s `.gitignore`.

### 2. Environment variables

You can also configure via plain env vars:

```bash
export GEMINI_API_KEY="your-gemini-key"
export GITMEUP_MODEL="gemini-2.5-flash"
```

### 3. CLI overrides (use sparingly)

The CLI accepts overrides:

```bash
gitmeup --model gemini-2.5-pro        # override model for this run only
gitmeup --api-key "you-can-but-should-not-add-your-key-here"     # override key (not recommended; leaks to history!)
```

> [!WARNING]
> For security, prefer `~/.gitmeup.env` or environment variables over `--api-key`.

### 4. Typical dev local use

Run this to use the fix in your installed CLI:

```bash
cd /home/you/gitmeup
source .venv/bin/activate

# reinstall updated code
python -m pip install . --force-reinstall

# verify resolved model from env loading
python - <<'PY'
import os
import gitmeup.cli as c
c.load_env()
print("GITMEUP_MODEL =", os.environ.get("GITMEUP_MODEL"))
PY

# optional: check if global file still has old model
grep -n '^GITMEUP_MODEL=' ~/.gitmeup.env .env 2>/dev/null || true

# test run (explicit model once, to avoid any ambiguity)
gitmeup --model gemini-2.5-flash-lite
gitmeup --model gemini-2.5-flash-lite --apply
```

## Usage

Run the tool from the root of any git repository with uncommitted changes:

```bash
gitmeup
```

Check the installed CLI version:

```bash
gitmeup -v
```

The tool performs the following checks:

* Verifies the current directory is a git repository.
* Checks `git status --porcelain` for changes.
* If changes exist, generates and displays **proposed commands**.

### Dry run (default)

```bash
gitmeup
```

Example output:

```bash
Proposed commands:

git add -- gitmeup/utils.py README.md
git commit -m 'docs: update README with configuration details'

Dry run: not executing commands. Re-run with --apply to execute.
```

![gitmeup run output example](./gitmeup-commit-example.png)

No commands are executed in this mode.

### Apply mode

To execute the proposed `git add` and `git commit` commands:

```bash
gitmeup --apply
```

`gitmeup` will:

* Print each command to stdout as it executes.
* Validate `git commit -m` headers against Conventional Commits (`<type>(scope): <description>`, scope optional).
* Auto-correct path casing for `git add`/`git rm`/`git mv` when a unique case-insensitive match exists in repo paths.
* Terminate immediately upon any command failure (non-zero exit code).
* Display the final git status upon completion:

```bash
Final git status:

## main...origin/main
 M src/api_client.py
?? tests/new_test.py

Review your history with:
  git log --oneline --graph --decorate -n 10
```

## Examples

Standard workflow using environment configuration:

```bash
# Review suggested batches
gitmeup

# Execute the plan
gitmeup --apply

# Verify history
git log --oneline --graph --decorate -n 10
```

Override the model for a specific run:

```bash
gitmeup --model gemini-2.5-flash-lite
```

## Behaviour

* **No pushing**: `gitmeup` never outputs `git push` or remote commands.
* **No invented files**: it only operates on files present in `git status` / `git diff`.
* **Strict quoting**: paths containing spaces, brackets, unicode, etc. are double-quoted; safe paths are not over-quoted.
* **Path casing safety**: command paths are normalized to real repo casing before execution; ambiguous matches fail with a clear error.
* **Conventional Commit guardrails**: commit headers must follow `<type>(scope): <description>` (scope optional) before any command is executed.
* **Atomic commits**: model is instructed to group changes into small, semantic batches (e.g. `refactor`, `docs`, `assets`), rather than one huge “misc” commit.

You still review and decide when to run `--apply`.

## License

This project is distributed under a **dual license** model.

1.  **Individuals & personal use**: Free to use under the terms of the **MIT License**.
2.  **Organizations & businesses**: A commercial license is required.

Please contact **ikrame@merakeen.com** for commercial licensing details.

See [`LICENSE`](./LICENSE) for the full legal text.

## Maintainer

Created & maintained by [@ikramagix](https://github.com/ikramagix).
