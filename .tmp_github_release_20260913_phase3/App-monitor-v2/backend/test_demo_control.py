from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from demo_control import DemoControlError, DemoControlManager

class DemoControlManagerTest(unittest.TestCase):
    @patch("demo_control.subprocess.Popen")
    def test_start_uses_fixed_oshima_location_and_stop_file(self, popen) -> None:
        process = MagicMock()
        process.pid = 123
        process.poll.return_value = None
        popen.return_value = process
        with tempfile.TemporaryDirectory() as directory:
            manager = DemoControlManager(Path(directory))
            status = manager.start(
                base_url="http://127.0.0.1:8000",
                scenario="af_emergency",
                loop=True,
            )
            command = popen.call_args.args[0]
            self.assertTrue(status["running"])
            self.assertIn("patient_help", command)
            self.assertIn("33.938502", command)
            self.assertIn("132.190863", command)
            self.assertIn("--stop-file", command)
            self.assertIn("--loop", command)
            manager._close_log()

    @patch("demo_control.subprocess.Popen")
    def test_second_start_is_rejected_while_running(self, popen) -> None:
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process
        with tempfile.TemporaryDirectory() as directory:
            manager = DemoControlManager(Path(directory))
            manager.start(base_url="http://localhost:8000", scenario="af_waiting", loop=False)
            with self.assertRaises(DemoControlError) as raised:
                manager.start(base_url="http://localhost:8000", scenario="af_waiting", loop=False)
            self.assertEqual(raised.exception.code, "already_running")
            manager._close_log()

if __name__ == "__main__":
    unittest.main()
