"""ブラウザからコンテスト用デモシナリオを安全に開始・停止する。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import RLock
from time import time
from uuid import uuid4

class DemoControlError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

class DemoControlManager:
    def __init__(self, backend_root: Path | None = None) -> None:
        self.backend_root = backend_root or Path(__file__).resolve().parent
        self.log_root = self.backend_root / "logs"
        self._process: subprocess.Popen | None = None
        self._stop_file: Path | None = None
        self._log_path: Path | None = None
        self._log_handle = None
        self._started_at_seconds: float | None = None
        self._scenario: str | None = None
        self._lock = RLock()

    def _refresh(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _remove_stop_file(self) -> None:
        if self._stop_file is not None and self._stop_file.is_file():
            self._stop_file.unlink()

    def status(self) -> dict:
        with self._lock:
            self._refresh()
            running = self._process is not None and self._process.poll() is None
            exit_code = None if self._process is None or running else self._process.returncode
            lines: list[str] = []
            if self._log_path is not None and self._log_path.is_file():
                lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            return {
                "schema_version": "1.0",
                "running": running,
                "pid": self._process.pid if running else None,
                "scenario": self._scenario,
                "started_at_seconds": self._started_at_seconds,
                "exit_code": exit_code,
                "recent_log": lines,
            }

    def start(self, *, base_url: str, scenario: str, loop: bool) -> dict:
        actions = {
            "af_waiting": "none",
            "af_unwell": "patient_unwell",
            "af_emergency": "patient_help",
        }
        if scenario not in actions:
            raise DemoControlError("invalid_scenario", "unsupported demo scenario")
        with self._lock:
            self._refresh()
            if self._process is not None and self._process.poll() is None:
                raise DemoControlError("already_running", "a demo scenario is already running")
            self.log_root.mkdir(parents=True, exist_ok=True)
            run_id = uuid4().hex[:8]
            self._stop_file = self.log_root / f"demo-control-{run_id}.stop"
            self._log_path = self.log_root / f"demo-control-{run_id}.log"
            self._log_handle = self._log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(self.backend_root / "contest_demo_scenario.py"),
                "--base-url",
                base_url,
                "--patient-action",
                actions[scenario],
                "--demo-latitude",
                "33.938502",
                "--demo-longitude",
                "132.190863",
                "--stop-file",
                str(self._stop_file),
            ]
            if loop:
                command.append("--loop")
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=self.backend_root,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            except OSError as error:
                self._close_log()
                raise DemoControlError("start_failed", str(error)) from error
            self._started_at_seconds = time()
            self._scenario = scenario
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._refresh()
            if self._process is None or self._process.poll() is not None:
                return self.status()
            if self._stop_file is not None:
                self._stop_file.write_text("stop\n", encoding="ascii")
            try:
                self._process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=4)
            finally:
                self._close_log()
                self._remove_stop_file()
            return self.status()

manager = DemoControlManager()
