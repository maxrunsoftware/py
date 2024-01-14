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
    log_format = "%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d [%(name)s]%(module)s:%(funcName)s] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format
    )

    RUNTIME_INFO.log_runtime_info(_log)

    window = Window()
    window.title = "Image Sorter"
    cmds = []
    cmd_classes = Command.__subclasses__()
    _log.info(f"Found {len(cmd_classes)} Commands")
    for i, cmd_class in enumerate(cmd_classes):
        _log.debug(f"cmd_classes[{i}]={cmd_class.__class__.__name__}")
        cmds.append(cmd_class(window))

    for cmd in cmds:
        cmd.add_to_window()
        window.layout.append([sg.HSep(pad=(0, 20))])

    # kdebug = "-MAIN.KDEBUG-"
    # kdebugtext = "-MAIN.KDEBUG_TEXT-"
    # window.layout.append([sg.Checkbox("Debug", key=kdebug, enable_events=True), sg.Push(), sg.Exit(size=(20, 1))])
    # window.layout.append([sg.Multiline(key=kdebugtext)])
    # window = sg.Window('Image Sorter', layout, font=font)
    # for cmd in cmds: cmd.window = window

    window.start()

    save_settings(cmds)


if __name__ == '__main__':
    main()
