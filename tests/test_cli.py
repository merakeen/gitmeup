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


if __name__ == "__main__":
    unittest.main()
