from pathlib import Path

from .command_base import *
from ..mrs.mrs_gui import *
from ..mrs.mrs_common import *


class TrimPrefix(CommandBase):
    def __init__(self, window: Window):
        super().__init__(window)
        self.title = "Trims Prefix Characters"

    def handle_scan(self, event: WindowEvent, directory: Path, is_recursive: bool):
        self._log.debug(f"Handling scan [{is_recursive=}]: {directory}")
        entries = fs_list(str(directory), recursive=is_recursive)

        for entry in entries:
            s = "D" if entry.is_dir() else " "
            s += " "
            s += entry.path
            print(s)
