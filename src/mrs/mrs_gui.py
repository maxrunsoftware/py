from __future__ import annotations
from typing import Dict, List, Set, Tuple
from PySimpleGUI import Element
from .mrs_common import *
import PySimpleGUI as sg
import logging

#_log = logger(__name__)


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
    NOT_SPECIFIED: str = "?"
    DELIMITER = "."
    PREFIX = "-"
    SUFFIX = "-"

    def __init__(self, parts: Union[str, List[str]]):
        super(WindowKey, self).__init__()
        pysimplegui_key = None
        if isinstance(parts, str): pysimplegui_key = parts
        elif isinstance(parts, list) and len(parts) == 1: pysimplegui_key = parts[0]
        if pysimplegui_key is not None and parts.startswith("-") and parts.startswith("-") and parts.endswith("-"):
            self._name = pysimplegui_key
        else:
            parts_list = parts if isinstance(parts, list) else [str(parts)]
            parts_split = (WindowKey.DELIMITER.join(parts_list)).split(WindowKey.DELIMITER)
            parts_list = list(filter_none(map(WindowKey.__clean_part, parts_split)))
            if not parts_list: raise ValueError(f"Invalid key {parts}")
            self._name = "-" + (".".join(parts_list)) + "-"

        self._name_casefold = self._name.casefold()
        self._hash = hash(self._name_casefold)

    @staticmethod
    def __clean_part(p: str) -> Optional[str]:
        p = trim(p)
        if p is None: return p
        while p is not None and p.startswith(WindowKey.PREFIX): p = trim(p[1:])  # remove first char
        while p is not None and p.endswith(WindowKey.SUFFIX): p = trim(p[:-1])  # remove last char
        return p

    @property
    def name(self) -> str: return self._name

    def __eq__(self, other):
        if isinstance(other, WindowKey):
            return self._hash == other._hash and self._name_casefold == other._name_casefold
        return NotImplemented

    def __hash__(self): return self._hash

    def __str__(self): return self.name

    def __repr__(self): return f"{self.class_name}({repr(self.name)})"


class WindowKeyBuilder:
    def __init__(self, name: str, keys_all: Optional[Set[WindowKey]] = None, parent: Optional[WindowKeyBuilder] = None):
        super(WindowKeyBuilder, self).__init__()
        self.keys_all = keys_all
        if self.keys_all is None and parent is not None: self.keys_all = parent.keys_all
        if self.keys_all is None: self.keys_all = set()
        self._name = name
        self._parent = parent
        self._parts = [name]
        if parent is not None: self._parts = parent._parts + [name]

    def __getitem__(self, name: str) -> WindowKeyBuilder:
        return WindowKeyBuilder(name, keys_all=self.keys_all, parent=self)

    def key(self, name) -> WindowKey:
        key = WindowKey(self._parts + [name])
        self.keys_all.add(key)
        return key


class WindowEvent:
    def __init__(self, window: Window, event: str, values: Dict[str, Any]):
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

    def __init__(self):
        super(Window, self).__init__()
        self.title: str = Window.TITLE_NOT_DEFINED
        self._subscribers: Dict[WindowKey, List[Callable[WindowEvent]]] = {}
        self._subscribers_exit: List[Callable[WindowEvent]] = []
        self.key_builder = WindowKeyBuilder(self.class_instance_name)
        self.font: str = "Ariel"
        self.font_size: int = 12
        self.theme: str = "dark"
        self.pysimplegui_window: Optional[sg.Window] = None
        self._window_size_x = -1
        self._window_size_y = -1

    def subscribe(self, key: WindowKey, handler: Callable[WindowEvent]):
        if key not in self._subscribers: self._subscribers[key] = []
        self._subscribers[key].append(handler)

    def subscribe_exit(self, handler: Callable[WindowEvent]):
        self._subscribers_exit.append(handler)

    def start(self):
        if self.title is None or self.title == Window.TITLE_NOT_DEFINED:
            raise ValueError(f"Attribute 'title' has not been set for window")

        sg.theme(self.theme)
        font = (self.font, self.font_size)
        sg.set_options(
            suppress_error_popups=True,
            suppress_raise_key_errors=False,
            suppress_key_guessing=True,
            warn_button_key_duplicates=True,
        )

        self.pysimplegui_window = window = sg.Window(self.title, self.layout, font=font)
        window.Finalize()
        window.BringToFront()
        self.check_window_size()

        while True:
            event, values = window.read()
            we = WindowEvent(self, event, values)
            self._log.debug(we)
            if we.event_raw in (sg.WIN_CLOSED, 'Exit'):
                self._log.debug("Exiting...")
                for subscriber in self._subscribers_exit: subscriber(we)
                break
            else:
                self.check_window_size()
                if we.event in self._subscribers:
                    subscribers = self._subscribers[we.event]
                    self._log.debug(f"Calling {len(subscribers)} subscribers for event {we}")
                    for i, subscriber in enumerate(subscribers):
                        self._log.debug(f"Calling subscribers[{i}] for event {we} -> {subscriber}")
                        subscriber(we)
                else:
                    self._log.warning(f"No subscriber for event {we}")

    def check_window_size(self):
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

    def __getitem__(self, key: Union[str, WindowKey]):
        k = str(key)
        try:
            return self.pysimplegui_window[k]
        except KeyError:
            allkeys = "\n".join(sorted([str(k) for k in self.pysimplegui_window.AllKeysDict.keys()], key=str.casefold))
            self._log.exception(f"Window does not have element '{k}' listing all keys:\n{allkeys}")
            raise


class WindowElement(ABC, ClassInfo, ClassLogging):
    def __init__(self, window: Window):
        super(WindowElement, self).__init__()
        self.window = window
        self.key_builder = window.key_builder[self.class_instance_name]
        self.layout: List = []

    def key(self, name: str) -> WindowKey: return self.key_builder.key(name)

    def subscribe_handler(self, keys: Union[WindowKey, List[WindowKey]], handler: Callable[WindowEvent]):
        keys = keys if isinstance(keys, list) else [keys]
        for key in keys:
            self.window.subscribe(key, handler)


class WindowElementDirSelect(WindowElement):
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


class WindowElementCollapsible(WindowElement):
    def __init__(
            self,
            window: Window,
            layout_items: List[Any],
            title_text: str = "TITLE",
            arrow_up: str = sg.SYMBOL_UP,
            arrow_down: str = sg.SYMBOL_DOWN,
            collapsed: bool = True
    ):
        super(WindowElementCollapsible, self).__init__(window)
        self.arrow_up = arrow_up
        self.arrow_down = arrow_down

        self.arrow_key = self.key("arrow")
        self.arrow = sg.Text((arrow_up if collapsed else arrow_down), enable_events=True, key=str(self.arrow_key))

        self.title_key = self.key("title")
        self.title = sg.Text(title_text, enable_events=True, key=str(self.title_key))

        self.section_column_key = self.key("section_column")
        self.section_column = sg.Column(
            layout_items,
            key=str(self.section_column_key),
            visible=not collapsed,
            # metadata=(self.arrow_down, self.arrow_up)
        )

        # self.section_pin = sg.pin(self.section_column)

        self.column_key = self.key("column")
        self.column = sg.Column([
            [self.arrow, self.title],
            [sg.pin(self.section_column)]
        ], key=str(self.column_key))

        self.layout = self.column
        self.subscribe_handler([self.arrow_key, self.title_key], self.collapse_handler)

    def collapse_handler(self, event: WindowEvent):
        sc = event.window[self.section_column_key]
        visible = sc.visible
        sc.update(visible=not sc.visible)
        arw = event.window[self.arrow_key]
        arw.update(self.arrow_up if visible else self.arrow_down)


