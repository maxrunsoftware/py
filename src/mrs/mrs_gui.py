from __future__ import annotations

import uuid
from typing import Dict, List, Set, Tuple
from PySimpleGUI import Element
from .mrs_common import *
import PySimpleGUI as sg
import logging

_log = logger(__name__)


class WindowColor(str, Enum):
    BLACK = "#000000"
    WHITE = "#FFFFFF"
    RED_LIGHT = "#F5B7B1"
    GREEN_LIGHT = "#ABEBC6"


class WindowLayout:
    def __init__(self):
        super(WindowLayout, self).__init__()
        self.layout: List[Any] = []


class WindowEventSubscriber(ABC):
    def __init__(self):
        super(WindowEventSubscriber, self).__init__()
        self.subscriber_keys: List[WindowKey] = []

    @abstractmethod
    def handle_event(self, event: WindowEvent): raise NotImplemented


class WindowKey(ClassInfo):
    def __init__(self, key: str | WindowKey | None = None) -> None:
        super(WindowKey, self).__init__()
        self._name = str(key) if key is not None else str(uuid.uuid4().hex)
        self._name_casefold = self._name.casefold()
        self._hash = hash(self._name_casefold)

    @property
    def name(self) -> str: return self._name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.__eq__(WindowKey(other))

        if isinstance(other, WindowKey):
            return self._hash == other._hash and self._name_casefold == other._name_casefold

        return NotImplemented

    def __hash__(self): return self._hash

    def __str__(self): return self.name

    def __repr__(self): return f"{self.class_name}({repr(self.name)})"




class WindowEvent:
    def __init__(self, window: Window, event: str | None, values: Dict[str, Any]):
        super(WindowEvent, self).__init__()
        self.window = window
        self.event_raw = event
        self.event = WindowKey(event) if event is not None else None
        self.values_raw = values
        vals: Dict[WindowKey, Any] = {}

        if values is not None:
            for k, v in values.items():
                wk = WindowKey(k)
                vals[wk] = v

        self.values = vals

    def __str__(self): return f"Event:{self.event}  Values:{self.values}"

    def __getitem__(self, key: WindowKey) -> Optional[Any]: return self.values[key] if key in self.values else None


class Window(WindowLayout, ClassInfo, ClassLogging):
    TITLE_NOT_DEFINED: str = "TITLE_NOT_DEFINED"
    FONT: str = "Ariel"
    FONT_SIZE: int = 12
    THEME: str = "dark"

    def __init__(self):
        super(Window, self).__init__()
        self.title: str = Window.TITLE_NOT_DEFINED
        self._subscribers: Dict[WindowKey, List[Callable[WindowEvent]]] = {}
        self._subscribers_exit: List[Callable[WindowEvent]] = []
        self.font: str = self.__class__.FONT
        self.font_size: int = self.__class__.FONT_SIZE
        self.theme: str = self.__class__.THEME
        self.pysimplegui_window: Optional[sg.Window] = None
        self._window_size_x = -1
        self._window_size_y = -1
        self._window_key_id_counter = 0

    def subscribe(self, key: str | WindowKey | [WindowKey], handler: Callable[WindowEvent]):
        keys: [WindowKey] = []
        if isinstance(key, str):
            keys.append(WindowKey(key))
        elif isinstance(key, WindowKey):
            keys.append(key)
        elif isinstance(key, list):
            keys = key

        for k in keys:
            if k not in self._subscribers:
                self._subscribers[k] = []

            self._subscribers[k].append(handler)

    def create_key_id(self) -> str:
        self._window_key_id_counter += 1
        return str(self._window_key_id_counter)

    def create_key(self, window_key_id: str, name: str) -> WindowKey:
        return WindowKey(f"-{window_key_id}:{name}-")

    def create_keys(self, *args) -> [WindowKey]:
        key_id = self.create_key_id()
        keys = []
        for arg in args:
            keys.append(self.create_key(key_id, arg))
        return keys

    def start(self):
        if self.title is None or self.title == Window.TITLE_NOT_DEFINED:
            raise ValueError(f"Attribute 'title' has not been set for window")

        sg.theme(self.theme)
        sg.set_options(
            suppress_error_popups=True,
            suppress_raise_key_errors=False,
            suppress_key_guessing=True,
            warn_button_key_duplicates=True,
        )

        self.pysimplegui_window = window = sg.Window(self.title, self.layout, font=(self.font, self.font_size))
        window.Finalize()
        window.BringToFront()
        self._check_window_size()

        continue_loop: bool = True
        while continue_loop:
            continue_loop = self._start_loop()

    def _start_loop(self) -> bool:
        event, values = self.pysimplegui_window.read()
        window_event = WindowEvent(self, event, values)
        self._log.debug(window_event)
        if window_event.event_raw in (sg.WIN_CLOSED, 'Exit'):
            self._log.debug("Exiting...")
            return False

        self._check_window_size()

        self._log.debug(f"Subscribers={len(self._subscribers.keys())}")
        for i, subscriber in enumerate(self._subscribers.keys()):
            self._log.debug(f"  Subscriber[{i}]: {subscriber}")


        if window_event.event in self._subscribers:
            subscribers = self._subscribers[window_event.event]
            self._log.debug(f"Calling {len(subscribers)} subscribers for event {window_event}")
            for i, subscriber in enumerate(subscribers):
                self._log.debug(f"Calling subscribers[{i}] for event {window_event} -> {subscriber}")
                subscriber(window_event)
        else:
            self._log.warning(f"No subscriber for event {window_event}")

        return True

    def _check_window_size(self):
        window_size_x, window_size_y = self.pysimplegui_window.size
        self._log.debug(f"Window size: {window_size_x} x {window_size_y}")
        window_size_x_old = self._window_size_x
        window_size_y_old = self._window_size_y
        if window_size_x > window_size_x_old or window_size_y > window_size_y_old:
            self._window_size_x = window_size_x
            self._window_size_y = window_size_y
            self._log.debug(f"Locking window size to {self._window_size_x} x {self._window_size_y}  screensize={sg.Window.get_screen_size()}")
            self.pysimplegui_window.set_min_size((window_size_x, window_size_y))

    @property
    def name(self) -> str: return f"{self.class_instance_name}({self.title})"

    def __getitem__(self, key: str | WindowKey):
        k = str(key)
        try:
            return self.pysimplegui_window[k]
        except KeyError:
            all_keys = "\n".join(sorted([str(k) for k in self.pysimplegui_window.AllKeysDict.keys()], key=str.casefold))
            self._log.exception(f"Window does not have element '{k}' listing all keys:\n{all_keys}")
            raise


class WindowElementDirSelect():
    def __init__(
            self,
            window: Window,
            label_text: str = "Directory",
            show_recursive: bool = False
    ):
        super(WindowElementDirSelect, self).__init__(window)
        self.label_key = self.key("label")
        self.label = sg.Text(
            label_text,
            key=str(self.label_key),
            size=(8, 1),
            justification="right",
        )

        self.input_key = self.key("input")
        self.input = sg.Input(
            key=str(self.input_key),
            enable_events=True,
            text_color=WindowColor.BLACK,
            background_color=WindowColor.WHITE,
        )

        self.browse_key = self.key("browse")
        self.browse = sg.FolderBrowse(
            key=str(self.browse_key),
            target=str(self.input_key),
        )

        self.recursive_key = self.key("recursive")
        self.recursive = sg.Checkbox(
            "Recursive",
            key=str(self.recursive_key),
            visible=show_recursive,
        )

        self.layout = [
            self.label,
            self.input,
            self.browse,
            self.recursive
        ]
        self.subscribe_handler(self.input_key, self.input_handler)

    def input_handler(self, event: WindowEvent):
        d = event[self.input_key]
        color: str = sg.theme_text_element_background_color()
        if trim(d) is not None:
            p = Path(d)
            color = WindowColor.GREEN_LIGHT if p.exists() and p.is_dir() else WindowColor.RED_LIGHT
        self._log.debug(f"Setting background color of '{self.input_key}' to {color}")
        event.window[self.input_key].update(background_color=color)

    def parse_event(self, event: WindowEvent) -> WindowElementDirSelectValues:
        is_recursive = event[self.recursive_key]
        if is_recursive is None: is_recursive = False

        directory_str = event[self.input_key]
        directory = None if directory_str is None else Path(directory_str)

        is_valid = False
        if directory is not None: is_valid = (directory.exists() and directory.is_dir())

        return WindowElementDirSelectValues(is_recursive=is_recursive, directory=directory, is_valid=is_valid)


class WindowElementDirSelectValues(NamedTuple):
    is_recursive: bool
    directory: Optional[Path]
    is_valid: bool


def create_column_collapsible(
        window: Window,
        layout_items: List[Any],
        title_text: str = "TITLE",
        arrow_up: str = sg.SYMBOL_UP,
        arrow_down: str = sg.SYMBOL_DOWN,
        collapsed: bool = True
) -> sg.Column:
    arrow_key, title_key, section_column_key, column_key = window.create_keys("arrow_key", "title_key", "section_column_key", "column_key")

    arrow = sg.Text((arrow_up if collapsed else arrow_down), enable_events=True, key=arrow_key)

    title = sg.Text(title_text, enable_events=True, key=str(title_key))

    section_column = sg.Column(
        layout_items,
        key=str(section_column_key),
        visible=not collapsed,
        # metadata=(self.arrow_down, self.arrow_up)
    )

    column = sg.Column([
        [arrow, title],
        [sg.pin(section_column)]
    ], key=column_key)

    def collapse_handler(event: WindowEvent):
        sc = event.window[section_column_key]
        visible = sc.visible
        sc.update(visible=not sc.visible)
        arrow_element = event.window[arrow_key]
        arrow_element.update(arrow_up if visible else arrow_down)

    window.subscribe([arrow_key, title_key, section_column_key], collapse_handler)
    return column
