import json
import logging
import os
import re
from abc import ABC, abstractmethod, ABCMeta
from pathlib import Path
from pprint import pprint, pformat
from typing import Optional

from mrs import *
import PySimpleGUI as sg
from mrs.mrs_gui import *

import logging

from src import cmd_gui

_log = logger(__name__)


class Command(ABC, ClassInfo, ClassLogging):
    def __init__(self, window: Window):
        super(Command, self).__init__()
        self.window = window
        self.key_builder = kb = window.key_builder[self.class_name]
        self.dir_select = WindowElementDirSelect(window, show_recursive=True)
        self.scan_key = kb.key("scan")
        self.scan = sg.Button("Scan", key=str(self.scan_key))
        self.section_content = [self.dir_select.layout, [sg.Push(), self.scan]]
        extras = self.elements_extra
        if extras is not None and len(extras) > 0: self.section_content.append(extras)

        # self.section_content = [sg.Text("Hello World")]
        self.section = WindowElementCollapsible(window, self.section_content, title_text=self.title_text)
        window.subscribe(self.scan_key, self.scan_handle)

    def scan_handle(self, event: WindowEvent):
        vals = self.dir_select.parse_event(event)
        self._log.debug(f"{vals=}")
        if not vals.is_valid: return
        self._log.debug(f"Scan:{vals.directory}  Recursive:{vals.is_recursive}")
        self.handle_scan(event, vals.directory, vals.is_recursive)

    @abstractmethod
    def handle_scan(self, event: WindowEvent, directory: Path, is_recursive: bool): raise NotImplemented

    @property
    def title_text(self): return re.sub(r'(?<![A-Z\W])(?=[A-Z])', ' ', self.class_name)  # https://stackoverflow.com/a/64834756

    @property
    def elements_extra(self) -> List: return []

    def add_to_window(self): self.window.layout.append([self.section.layout])


class RenameByExif(Command):
    def __init__(self, window: Window):
        super().__init__(window)
        self.title = "Rename by EXIF"

    def handle_scan(self, event: WindowEvent, directory: Path, is_recursive: bool):
        self._log.debug(f"Handling scan [{is_recursive=}]: {directory}")
        entries = fs_list(str(directory), recursive=is_recursive)

        for entry in entries:
            s = "D" if entry.is_dir() else " "
            s += " "
            s += entry.path
            print(s)


def save_settings(commands: list[Command]):
    _log.debug("Saving settings")


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(module)s.%(funcName)s %(filename)s:%(lineno)d [%(name)s]: %(message)s"
    )

    RUNTIME_INFO.log_runtime_info(_log)
    cmd_gui.run2()



if __name__ == '__main__':
    main()