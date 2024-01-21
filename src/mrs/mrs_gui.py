from __future__ import annotations

import uuid

# noinspection PyPep8Naming
import PySimpleGUI as sg

from .mrs_common import *

_log = logger(__name__)


class WindowColor(str, Enum):
    BLACK = '#000000'
    WHITE = '#FFFFFF'
    RED_LIGHT = '#F5B7B1'
    GREEN_LIGHT = '#ABEBC6'
    YELLOW_LIGHT = '#F4F5B1'


class WindowKey(ClassInfo):
    DELIMITER: str = '.'

    def __init__(self, *args) -> None:
        super(WindowKey, self).__init__()
        items = trim(xstr(args), exclude_none=True)
        parts = [items] if isinstance(items, str) else items
        if len(parts) == 0:
            parts.append(f"UNKNOWN[{uuid.uuid4().hex}]")

        self._name = self.__class__.DELIMITER.join(parts)
        self._name_casefold = self._name.casefold()
        self._hash = hash(self._name_casefold)

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other):
        if not isinstance(other, WindowKey):
            return self.__eq__(WindowKey(other))
        return self._hash == other._hash and self._name_casefold == other._name_casefold

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.class_name}({repr(self.name)})"

    def sub(self, *args) -> WindowKey:
        return WindowKey(self, *args)


class WindowEvent:
    def __init__(self, window: Window, event: str | None, values: Dict[str, Any]):
        super(WindowEvent, self).__init__()
        self.window = window
        self.event_raw = event
        self.is_exit = True if event in (sg.WIN_CLOSED, 'Exit') else False
        self.event = WindowKey(event) if event is not None else None
        self.values_raw = values

        vals: Dict[WindowKey, Any] = {}
        if values is not None:
            for k, v in values.items():
                wk = WindowKey(k)
                vals[wk] = v
        self.values = vals

    def __str__(self):
        return f"Event:{self.event}  Values:{self.values}"

    def __getitem__(self, key: WindowKey) -> Optional[Any]:
        return self.values[key] if key in self.values else None


class Window(ClassInfo, ClassLogging):

    def __init__(self):
        super(Window, self).__init__()
        self.layout: List[Any] = []
        self.title: str | None = None
        self._subscribers: Dict[WindowKey, List[Callable[WindowEvent]]] = {}
        # self._subscribers_exit: List[Callable[WindowEvent]] = []
        self.font: str = 'Ariel'
        self.font_size: int = 12
        self.theme: str = 'dark'
        self._pysimplegui_window: sg.Window | None = None
        self._window_size_x = -1
        self._window_size_y = -1
        self._key: WindowKey = WindowKey(f"{self._class_name}[{self._class_instance_id}]")

    @property
    def key(self) -> WindowKey:
        return self._key

    def subscribe(self, key: WindowKey | [WindowKey], handler: Callable[WindowEvent]):
        keys: [WindowKey] = key if isinstance(key, list) else [key]
        for k in keys:
            if k not in self._subscribers:
                self._subscribers[k] = []
            self._log.debug(f"Adding subscriber for event {k} -> {handler}")
            self._subscribers[k].append(handler)

    def start(self):
        if self.title is None:
            self.title = ' '.join(split_on_capital(self.class_name))

        sg.theme(self.theme)
        sg.set_options(
            suppress_error_popups=True,
            suppress_raise_key_errors=False,
            suppress_key_guessing=True,
            warn_button_key_duplicates=True,
        )

        self._pysimplegui_window = w = sg.Window(
            title=self.title,
            layout=self.layout,
            font=(self.font, self.font_size),
        )
        w.Finalize()
        w.BringToFront()
        self._check_window_size()

        continue_loop: bool = True
        while continue_loop:
            continue_loop = self._start_loop()

    def _start_loop(self) -> bool:
        event, values = self._pysimplegui_window.read()
        window_event = WindowEvent(self, event, values)
        self._log.debug(window_event)
        if window_event.is_exit:
            self._log.debug('Exiting...')
            return False

        self._check_window_size()

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
        window_size_x, window_size_y = self._pysimplegui_window.size
        self._log.debug(f"Window size: {window_size_x} x {window_size_y}")
        window_size_x_old = self._window_size_x
        window_size_y_old = self._window_size_y
        if window_size_x > window_size_x_old or window_size_y > window_size_y_old:
            self._window_size_x = window_size_x
            self._window_size_y = window_size_y
            self._log.debug(
                f"Locking window size to {self._window_size_x} x {self._window_size_y}  screensize="
                f"{sg.Window.get_screen_size()}"
            )
            self._pysimplegui_window.set_min_size((window_size_x, window_size_y))

    @property
    def name(self) -> str:
        return f"{self.class_instance_name}({self.title})"

    def __getitem__(self, key: WindowKey):
        k = key
        try:
            return self._pysimplegui_window[k]
        except KeyError:
            # noinspection PyTypeChecker
            all_keys = "\n".join(
                sorted(
                    [f"{kk}  ({type(kk).__name__})" for kk in self._pysimplegui_window.AllKeysDict.keys()],
                    key=str.casefold
                    )
            )
            self._log.exception(
                f"Window does not have element '{k}  ({type(k).__name__})' listing all keys:\n{all_keys}"
                )
            raise


class WindowElementBrowseDirEvent(NamedTuple):
    key: WindowKey
    is_recursive: bool
    directory: str
    directory_path: Path | None
    is_valid: bool

    @staticmethod
    def create(
        key: WindowKey, directory: str | Path | None = None, is_recursive: bool | None = False
    ) -> WindowElementBrowseDirEvent:
        if is_recursive is None:
            is_recursive = False
        d: str = ''
        dp: Path | None = None
        if trim(directory) is not None:
            if isinstance(directory, Path):
                d = str(directory)
                dp = directory
            elif isinstance(directory, str):
                d = directory
                dp = Path(directory)
        iv: bool = dp is not None and dp.exists() and dp.is_dir()
        return WindowElementBrowseDirEvent(
            key=key,
            is_recursive=is_recursive,
            directory=d,
            directory_path=dp,
            is_valid=iv,
        )


def window_element_browse_dir_create(
    window: Window,
    key_browse_dir: WindowKey,
    change_callback: Callable[WindowElementBrowseDirEvent],
    label_text: str = 'Directory',
    show_recursive_checkbox: bool = False,
    default_directory: str | Path | None = None,
    default_recursive_checked: bool = False,
) -> list[sg.Element]:
    k = key_browse_dir

    default_directory_str: str | None = None
    if default_directory is None:
        default_directory_str = None

    if isinstance(default_directory, Path):
        default_directory_str = str(default_directory)
    elif isinstance(default_directory, str) and trim(default_directory) is not None:
        default_directory_str = default_directory

    def get_input_background_color(browse_dir_event: WindowElementBrowseDirEvent) -> str:
        color = sg.theme_text_element_background_color()
        if trim(browse_dir_event.directory) is not None:
            color = WindowColor.GREEN_LIGHT if browse_dir_event.is_valid else WindowColor.RED_LIGHT
        return color

    label_key = k.sub('label')
    label = sg.Text(
        label_text,
        key=label_key,
        size=(8, 1),
        justification='right',
    )

    input_key = k.sub('input')
    # noinspection PyShadowingBuiltins
    input = sg.Input(
        key=input_key,
        enable_events=True,
        text_color=WindowColor.BLACK,
        background_color=get_input_background_color(
            WindowElementBrowseDirEvent.create(k, default_directory_str, default_recursive_checked)
        ),
        default_text='' if default_directory_str is None else default_directory_str,
    )

    browse_key = k.sub('browse')
    browse = sg.FolderBrowse(
        key=browse_key,
        target=str(input_key),  # https://github.com/PySimpleGUI/PySimpleGUI/issues/6260
        initial_folder=default_directory_str,
    )

    recursive_key = k.sub('recursive')
    recursive = sg.Checkbox(
        text='Recursive',
        key=recursive_key,
        visible=show_recursive_checkbox,
        default=default_recursive_checked,
    )

    layout = [label, input, browse, recursive]

    def input_changed_handler(event: WindowEvent):
        d = event[input_key]
        r = False if not show_recursive_checkbox else bool_parse(trim(xstr(event[recursive_key])) or False)
        browse_dir_event = WindowElementBrowseDirEvent.create(k, d, r)
        color = get_input_background_color(browse_dir_event)
        _log.debug(f"Setting background color of '{input_key}' to {color}")
        event.window[input_key].update(background_color=color)
        change_callback(browse_dir_event)

    window.subscribe([input_key, recursive_key], input_changed_handler)

    return layout


def window_element_column_collapsible_create(
    window: Window,
    key_column: WindowKey,
    layout_items: list[Any],
    title_text: str,
    arrow_up: str = sg.SYMBOL_UP,
    arrow_down: str = sg.SYMBOL_DOWN,
    collapsed: bool = True,
) -> sg.Column:
    k = key_column

    arrow_key = k.sub('arrow')
    arrow = sg.Text(
        text=(arrow_up if collapsed else arrow_down),
        enable_events=True,
        key=arrow_key,
    )

    title_key = k.sub('title')
    title = sg.Text(
        text=title_text,
        enable_events=True,
        key=title_key,
    )

    section_column_key = k.sub('section_column')
    section_column = sg.Column(
        layout=layout_items,
        key=section_column_key,
        visible=not collapsed,
        # metadata=(self.arrow_down, self.arrow_up),
    )

    column_key = k.sub('column_key')
    column = sg.Column(
        layout=[[arrow, title], [sg.pin(section_column)]],
        key=column_key,
    )

    def collapse_handler(event: WindowEvent):
        sc = event.window[section_column_key]
        is_visible = sc.visible
        sc.update(visible=not sc.visible)
        arrow_element = event.window[arrow_key]
        arrow_element.update(arrow_up if is_visible else arrow_down)

    window.subscribe([arrow_key, title_key, section_column_key], collapse_handler)

    return column
