from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import List

from src.mrs.mrs_common import ClassInfo, ClassLogging
from src.mrs.mrs_gui import Window, WindowElementDirSelect, WindowElementCollapsible, WindowEvent

import PySimpleGUI as sg


class CommandBase(ABC, ClassInfo, ClassLogging):
    def __init__(self, window: Window):
        super(CommandBase, self).__init__()

        self.window = window
        self.key_builder = kb = window.key_builder[self.class_name]
        self.dir_select = WindowElementDirSelect(window, show_recursive=True)
        self.scan_key = kb.key("scan")
        self.scan = sg.Button("Scan", key=str(self.scan_key))
        self.section_content = [self.dir_select.layout, [sg.Push(), self.scan]]
        extras = self.elements_extra
        if extras is not None and len(extras) > 0:
            self.section_content.append(extras)

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
