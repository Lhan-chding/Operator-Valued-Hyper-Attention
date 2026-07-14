import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "scripts" / "run_epoch2_phase2_handoff.sh"
SOURCE_COMMIT = "a" * 40
TARGET_COMMIT = "b" * 40
CHECKPOINT_SHA256 = (
    "b448804bb1af6fa688887f0f2454625edbeeae4e868bc95620e3e6413581051a"
)


class Epoch2Phase2HandoffTests(unittest.TestCase):
    def _arguments(self, parent: Path, python: Path) -> tuple[str, ...]:
        return (
            "--source-work-dir", str(parent / "source"),
            "--runtime-project-dir", str(parent / "runtime"),
            "--expected-source-project-commit", SOURCE_COMMIT,
            "--target-project-commit", TARGET_COMMIT,
            "--mmdet-root", str(parent / "mmdetection"),
            "--data-root", str(parent / "data"),
            "--checkpoint", str(parent / "official.pth"),
            "--checkpoint-sha256", CHECKPOINT_SHA256,
            "--bert-root", str(parent / "bert"),
            "--phase1-work-root", str(parent / "phase1"),
            "--phase2-work-root", str(parent / "phase2"),
            "--python", str(python),
            "--phase1-gpu", "4",
            "--phase2-gpu", "5",
            "--phase1-port", "29627",
            "--phase2-port", "29628",
            "--per-device-batch", "8",
            "--seed", "2026",
        )

    def test_dry_run_prints_exact_migration_and_fail_closed_handoff(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            result = subprocess.run(
                ("bash", str(HANDOFF), *self._arguments(parent, Path("python3")),
                 "--dry-run"),
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        source = HANDOFF.read_text(encoding="utf-8")
        self.assertIn(
            '"${SCRIPT_DIR}/migrate_epoch_resume.py"', source)
        self.assertIn(
            '"${SCRIPT_DIR}/run_migrated_resume.py"', source)
        self.assertIn("--expected-source-checkpoint epoch_1.pth", result.stdout)
        self.assertIn("--resume --stop-after-epoch2", result.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=4", result.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=5", result.stdout)
        self.assertIn("epoch_2.pth", result.stdout)
        self.assertLess(
            result.stdout.index("migrate_epoch_resume.py"),
            result.stdout.index("run_migrated_resume.py"),
        )
        self.assertLess(
            result.stdout.index("run_migrated_resume.py"),
            result.stdout.index("phase2_preflight.py"),
        )
        self.assertLess(
            result.stdout.index("phase2_preflight.py"),
            result.stdout.index("run_phase2_full_server.sh"),
        )

        self.assertIn('export PATH="/usr/bin:/bin"', source)

    def test_phase1_failure_preserves_status_and_never_launches_phase2(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            runtime_scripts = parent / "runtime" / "scripts"
            runtime_scripts.mkdir(parents=True)
            for name in (
                "migrate_epoch_resume.py",
                "run_migrated_resume.py",
                "run_phase2_full_server.sh",
            ):
                (runtime_scripts / name).write_text("# reviewed placeholder\n")
            log = parent / "calls.log"
            fake_python = parent / "python"
            fake_python.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                printf '%s\\n' "$1" >> {str(log)!r}
                if [[ "$1" == *run_migrated_resume.py ]]; then
                  exit 7
                fi
                exit 0
            """))
            fake_python.chmod(0o700)
            environment = dict(os.environ)
            environment["HANDOFF_TEST_LOG"] = str(log)

            result = subprocess.run(
                ("bash", str(HANDOFF), *self._arguments(parent, fake_python)),
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                env=environment,
                check=False,
            )

            self.assertEqual(result.returncode, 7, result.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("migrate_epoch_resume.py", calls)
            self.assertIn("run_migrated_resume.py", calls)
            self.assertNotIn("run_phase2_full_server.sh", calls)

    def test_migration_failure_preserves_status_and_stops_handoff(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            runtime_scripts = parent / "runtime" / "scripts"
            runtime_scripts.mkdir(parents=True)
            for name in (
                "migrate_epoch_resume.py",
                "run_migrated_resume.py",
                "run_phase2_full_server.sh",
            ):
                (runtime_scripts / name).write_text("# reviewed placeholder\n")
            log = parent / "calls.log"
            fake_python = parent / "python"
            fake_python.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                printf '%s\\n' "$1" >> {str(log)!r}
                exit 6
            """))
            fake_python.chmod(0o700)

            result = subprocess.run(
                ("bash", str(HANDOFF), *self._arguments(parent, fake_python)),
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )

            self.assertEqual(result.returncode, 6, result.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("migrate_epoch_resume.py", calls)
            self.assertNotIn("run_migrated_resume.py", calls)
            self.assertNotIn("run_phase2_full_server.sh", calls)

    def test_missing_epoch2_stops_before_phase2(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            runtime_scripts = parent / "runtime" / "scripts"
            runtime_scripts.mkdir(parents=True)
            for name in (
                "migrate_epoch_resume.py",
                "run_migrated_resume.py",
                "run_phase2_full_server.sh",
            ):
                (runtime_scripts / name).write_text("# reviewed placeholder\n")
            log = parent / "calls.log"
            fake_python = parent / "python"
            fake_python.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                printf '%s\\n' "$1" >> {str(log)!r}
                exit 0
            """))
            fake_python.chmod(0o700)

            result = subprocess.run(
                ("bash", str(HANDOFF), *self._arguments(parent, fake_python)),
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("without exact epoch_2.pth", result.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("migrate_epoch_resume.py", calls)
            self.assertIn("run_migrated_resume.py", calls)
            self.assertNotIn("run_phase2_full_server.sh", calls)

    def test_phase2_failure_is_returned_to_the_caller(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            runtime_scripts = parent / "runtime" / "scripts"
            runtime_scripts.mkdir(parents=True)
            log = parent / "calls.log"
            epoch2 = (
                parent / "phase1" / "refcoco" / "rqgo" / "seed_2026"
                / "epoch_2.pth"
            )
            for name in ("migrate_epoch_resume.py", "run_migrated_resume.py"):
                (runtime_scripts / name).write_text("# reviewed placeholder\n")
            (runtime_scripts / "run_phase2_full_server.sh").write_text(
                textwrap.dedent(f"""\
                    printf '%s\\n' "$0" >> {str(log)!r}
                    exit 9
                """),
            )
            fake_python = parent / "python"
            fake_python.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                printf '%s\\n' "$1" >> {str(log)!r}
                if [[ "$1" == *run_migrated_resume.py ]]; then
                  mkdir -p {str(epoch2.parent)!r}
                  printf '%s\\n' 'complete epoch two' > {str(epoch2)!r}
                fi
                exit 0
            """))
            fake_python.chmod(0o700)

            result = subprocess.run(
                ("bash", str(HANDOFF), *self._arguments(parent, fake_python)),
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )

            self.assertEqual(result.returncode, 9, result.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertIn("migrate_epoch_resume.py", calls)
            self.assertIn("run_migrated_resume.py", calls)
            self.assertIn("run_phase2_full_server.sh", calls)

    def test_successful_epoch2_resume_launches_phase2_with_exact_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            parent = Path(directory)
            runtime_scripts = parent / "runtime" / "scripts"
            runtime_scripts.mkdir(parents=True)
            for name in ("migrate_epoch_resume.py", "run_migrated_resume.py"):
                (runtime_scripts / name).write_text("# reviewed placeholder\n")
            log = parent / "calls.log"
            epoch2 = parent / "phase1/refcoco/rqgo/seed_2026/epoch_2.pth"
            phase2 = runtime_scripts / "run_phase2_full_server.sh"
            phase2.write_text(textwrap.dedent("""\
                #!/usr/bin/env bash
                printf 'phase2 %s\\n' "$*" >> "$HANDOFF_TEST_LOG"
            """))
            fake_python = parent / "python"
            fake_python.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                printf 'python %s\\n' "$*" >> "$HANDOFF_TEST_LOG"
                if [[ "$1" == *run_migrated_resume.py ]]; then
                  mkdir -p {str(epoch2.parent)!r}
                  : > {str(epoch2)!r}
                fi
                exit 0
            """))
            fake_python.chmod(0o700)
            environment = dict(os.environ)
            environment["HANDOFF_TEST_LOG"] = str(log)

            result = subprocess.run(
                ("bash", str(HANDOFF), *self._arguments(parent, fake_python)),
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                env=environment,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text(encoding="utf-8")
            self.assertLess(
                calls.index("migrate_epoch_resume.py"),
                calls.index("run_migrated_resume.py"),
            )
            self.assertLess(
                calls.index("run_migrated_resume.py"), calls.index("phase2 "))
            self.assertIn(f"--epoch2-checkpoint {epoch2}", calls)
            self.assertIn(
                f"--expected-source-project-commit {TARGET_COMMIT}", calls)


if __name__ == "__main__":
    unittest.main()
