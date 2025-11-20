# gitmeup

`gitmeup` is a lightweight command-line tool that automates Conventional Commit generation using an LLM.
It analyses your working tree using `git diff --stat`, `git status`, and a filtered `git diff`, then produces
atomically grouped `git add` and `git commit` commands with strict path-quoting rules.

This improves commit hygiene in large or fast-moving repositories while reducing cognitive overhead.

## Features

- Extracts:
  - `git diff --stat`
  - `git status --short`
  - `git diff` excluding image and binary noise
- Sends structured context to an LLM
- Produces:
  - Precise Conventional Commits
  - Batched `git add` / `git commit` sequences
  - Proper double-quote path escaping
- Dry-run or apply mode
- Minimal overhead, only Python and git required

## Installation

### Local development (editable)

```bash
git clone https://github.com/ikramagix/gitmeup
cd gitmeup

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

### System-wide (user environment)

```bash
pip install git+https://github.com/ikramagix/gitmeup
```

Ensure `GEMINI_API_KEY` is configured:

```bash
export GEMINI_API_KEY="your-key"
```

You also need a model with:

```bash
export GITMEUP_MODEL="a gemini model"
```

## Usage

From any git repository with uncommitted changes:

```bash
gitmeup
```

Dry-run prints proposed commands. To apply staged operations:

```bash
gitmeup --apply
```

Specify a model:

```bash
gitmeup --model gpt-4.1
```

Specify an API key manually:

```bash
gitmeup --api-key sk-....
```

## Example

```bash
gitmeup
```

Sample output:

```bash
git add src/app/main.py
git add src/app/utils/helpers.py
git commit -m "refactor(core): clean helper logic"
```

## License

MIT License. See `LICENSE` for details.

## Maintainer

Created and maintained by **@ikramagix**.