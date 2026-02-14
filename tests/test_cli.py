import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from gitmeup import cli


class ParseCommandsTests(unittest.TestCase):
    def test_parse_commands_supports_multiline_quoted_arguments(self):
        block = 'git commit -m "feat: start\ncontinue"'

        commands = cli.parse_commands(block)

        self.assertEqual(commands, [["git", "commit", "-m", "feat: start\ncontinue"]])

    def test_parse_commands_ignores_comments_and_shell_prompt_prefix(self):
        block = """
# generated commands
$ git add -- foo.py
$ git commit -m "fix: update parser"
"""

        commands = cli.parse_commands(block)

        self.assertEqual(
            commands,
            [
                ["git", "add", "--", "foo.py"],
                ["git", "commit", "-m", "fix: update parser"],
            ],
        )

    def test_parse_commands_reports_unterminated_quote(self):
        with self.assertRaisesRegex(ValueError, "Unterminated quoted string"):
            cli.parse_commands('git commit -m "broken')


class MainErrorHandlingTests(unittest.TestCase):
    @patch("gitmeup.cli.load_env")
    @patch("gitmeup.cli.ensure_repo")
    @patch("gitmeup.cli.run_git")
    @patch("gitmeup.cli.collect_context", return_value=("stat", "status", "diff"))
    @patch("gitmeup.cli.call_llm", return_value='```bash\ngit commit -m "broken\n```')
    def test_main_handles_parse_errors_without_traceback(
        self,
        _mock_call_llm,
        _mock_collect_context,
        mock_run_git,
        _mock_ensure_repo,
        _mock_load_env,
    ):
        # First run_git call is "status --porcelain". Keep it non-empty to proceed.
        mock_run_git.return_value = " M file.py\n"

        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as exit_ctx:
                cli.main(["--api-key", "test-key"])

        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("failed to parse bash commands", err.getvalue())

    @patch("gitmeup.cli.load_env")
    def test_main_version_flag_prints_version(self, _mock_load_env):
        out = io.StringIO()
        with redirect_stdout(out):
            with self.assertRaises(SystemExit) as exit_ctx:
                cli.main(["-v"])

        self.assertEqual(exit_ctx.exception.code, 0)
        self.assertRegex(out.getvalue().strip(), r"^gitmeup \d+\.\d+\.\d+$")


class RunCommandsValidationTests(unittest.TestCase):
    @patch("gitmeup.cli.run_git")
    def test_run_commands_corrects_case_mismatched_add_paths(self, mock_run_git):
        mock_run_git.side_effect = [
            "components/WebGLDisabledPopup.tsx\0",
            " M components/WebGLDisabledPopup.tsx\0",
        ]
        commands = [
            ["git", "add", "--", "Components/WebGLDisabledPopup.tsx"],
            ["git", "commit", "-m", "fix(ui): update popup behavior"],
        ]

        out = io.StringIO()
        with redirect_stdout(out):
            cli.run_commands(commands, apply=False)

        output = out.getvalue()
        self.assertIn("Adjusted path casing", output)
        self.assertIn("git add -- components/WebGLDisabledPopup.tsx", output)

    def test_run_commands_rejects_non_conventional_commit_headers(self):
        commands = [["git", "commit", "-m", "update parser behavior"]]

        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as exit_ctx:
                cli.run_commands(commands, apply=False)

        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("invalid Conventional Commit header", err.getvalue())

    def test_run_commands_rejects_generic_scope_name(self):
        commands = [["git", "commit", "-m", "refactor(gitmeup): improve parser validation"]]

        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as exit_ctx:
                cli.run_commands(commands, apply=False)

        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("scope 'gitmeup' is too generic", err.getvalue())

    def test_run_commands_rejects_empty_commit_header(self):
        commands = [["git", "commit", "-m", ""]]

        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as exit_ctx:
                cli.run_commands(commands, apply=False)

        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("commit header is empty", err.getvalue())

    @patch("gitmeup.cli.run_git")
    def test_run_commands_rejects_scoped_commit_with_multiple_areas(self, mock_run_git):
        mock_run_git.side_effect = [
            "README.md\0tests/test_cli.py\0",
            " M README.md\0 M tests/test_cli.py\0",
        ]
        commands = [
            ["git", "add", "--", "README.md", "tests/test_cli.py"],
            ["git", "commit", "-m", "chore(tests): sync docs and tests"],
        ]

        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as exit_ctx:
                cli.run_commands(commands, apply=False)

        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("spans multiple top-level areas", err.getvalue())

    @patch("gitmeup.cli.run_git")
    def test_run_commands_allows_unscoped_commit_with_multiple_areas(self, mock_run_git):
        mock_run_git.side_effect = [
            "README.md\0tests/test_cli.py\0",
            " M README.md\0 M tests/test_cli.py\0",
        ]
        commands = [
            ["git", "add", "--", "README.md", "tests/test_cli.py"],
            ["git", "commit", "-m", "chore: sync docs and tests"],
        ]

        out = io.StringIO()
        with redirect_stdout(out):
            cli.run_commands(commands, apply=False)

        self.assertIn("git commit -m 'chore: sync docs and tests'", out.getvalue())

    @patch("gitmeup.cli.run_git")
    def test_run_commands_fails_for_ambiguous_case_corrections(self, mock_run_git):
        mock_run_git.side_effect = [
            "components/Popup.tsx\0components/popup.tsx\0",
            "",
        ]
        commands = [["git", "add", "--", "components/POPUP.tsx"]]

        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as exit_ctx:
                cli.run_commands(commands, apply=False)

        self.assertEqual(exit_ctx.exception.code, 1)
        self.assertIn("Ambiguous case-insensitive match", err.getvalue())


if __name__ == "__main__":
    unittest.main()
