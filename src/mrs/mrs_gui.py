#  Copyright (c) 2024 Max Run Software (dev@maxrunsoftware.com)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

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
    BLUE_LIGHT = '#ADD8E6'
    BUTTON_DISABLED_FOREGROUND = '#666666'
    BUTTON_DISABLED_BACKGROUND = '#CCCCCC'
    BUTTON_DISABLED_BORDER = '#999999'


class WindowKey(ClassInfo):

    # region attributes

    DELIMITER: str = '.'

    _keys: dict[(str, ...), WindowKey | None] = dict()

    # endregion attributes

    # region @classmethod

    @classmethod
    def root_keys(cls) -> list[WindowKey]:
        return [x for x in sorted(key for key in cls._keys if len(key) == 1)]

    @classmethod
    def all_keys(cls) -> list[WindowKey]:
        return [x for x in sorted(key for key in cls._keys)]

    @classmethod
    def all_keys_str(cls) -> str:
        if len(cls._keys) == 0:
            return '!! No Keys Created !!'

        items = [(('  ' * (len(x) - 1)) + x.name) for x in cls.all_keys()]
        return '\n'.join(items)

    @classmethod
    def clear_keys(cls) -> int:
        key_count = len(cls._keys)
        cls._keys.clear()
        return key_count

    # endregion @classmethod

    # region init

    def __init__(self, *args) -> None:
        super(WindowKey, self).__init__()
        assert isinstance(args, tuple)

        def iterable_flatten_parser(item):
            return item.parts if item is not None and isinstance(item, WindowKey) else item

        parts: list[str] = trim(xstr(iterable_flatten(args, iterable_flatten_parser)), exclude_none=True)

        if len(parts) == 0:
            raise ValueError('No name or key parts provided')

        self._parts: tuple[str, ...] = tuple(parts)
        parts_casefold = tuple([x.casefold() for x in parts])
        self._parts_casefold: tuple[str, ...] = parts_casefold
        self._hash = hash(parts_casefold)

        parent_parts_casefold = None if len(parts) < 2 else parts_casefold[:-1]
        self._parent_parts_casefold: tuple[str, ...] | None = parent_parts_casefold
        self.__class__._keys[parts_casefold] = self

        if parent_parts_casefold is not None and parent_parts_casefold not in self.__class__._keys:
            WindowKey(parent_parts_casefold)  # force parent key creation

    # endregion init

    # region method

    def is_descendant_of(self, parent: WindowKey) -> bool:
        if len(parent) >= len(self):
            return False
        current = self
        while current is not None:
            if current == parent:
                return True
            current = current.parent
        return False

    def child(self, *args) -> WindowKey:
        return WindowKey(self, *args)

    # endregion method

    # region @property

    @property
    def children(self) -> Iterable[WindowKey]:
        self_len = len(self)
        self_parts = self._parts_casefold
        for item in sorted(
                child_key for child_key in self.__class__._keys.values() if
                len(child_key) == self_len + 1 and
                child_key._parent_parts_casefold == self_parts
        ):
            yield item

    @property
    def parent(self) -> WindowKey | None:
        parent_parts_casefold = self._parent_parts_casefold
        if parent_parts_casefold is None:
            return None
        parent = self.__class__._keys.get(parent_parts_casefold)
        if parent is None:
            self.__class__._keys[parent_parts_casefold] = parent = WindowKey(parent_parts_casefold)
        return parent

    @property
    def name(self) -> str:
        return self._parts[-1]

    @property
    def parts(self) -> tuple:
        return self._parts

    # endregion @property

    # region override

    def __len__(self):
        return len(self._parts)

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, str):
            other = trim(other)
            if other is None:
                return False
        if not isinstance(other, WindowKey):
            return self.__eq__(WindowKey(other))
        return self._hash == other._hash and self._parts_casefold == other._parts_casefold

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self.DELIMITER.join(self._parts)

    def __repr__(self):
        return f"{self.class_name}({self._parts})"

    def __lt__(self, other):
        if other is None:
            return False
        if not isinstance(other, WindowKey):
            return self.__lt__(WindowKey(other))
        parts_x = self._parts_casefold
        parts_y = other._parts_casefold
        len_x = len(parts_x)
        len_y = len(parts_y)
        for i in range(min(len_x, len_y)):
            if parts_x[i] < parts_y[i]:
                return True
            if parts_x[i] > parts_y[i]:
                return False
        return len_x < len_y

    # endregion override


class WindowEvent:

    # region init

    def __init__(self, window: Window, event: WindowKey | str | None, values: Dict[str, Any]):
        super(WindowEvent, self).__init__()
        self.window = window
        self.event_raw = event
        self.is_exit = True if event in (sg.WIN_CLOSED, 'Exit') else False

        def parse_key(s) -> WindowKey | None:
            if s is None:
                return None
            elif isinstance(s, WindowKey):
                return s
            elif not isinstance(s, str):
                s = trim(str(s))
            if s is None:
                return None
            return WindowKey(s)

        self.key: WindowKey | None = parse_key(event)
        self.values_raw = values

        vals: Dict[WindowKey, Any] = {}
        if values is not None:
            for k, v in values.items():
                wk = parse_key(k)
                if wk is not None:
                    vals[wk] = v
                else:
                    logger(cls=type(self)).warning(f"Item in event data not attached to WindowKey {k}={v}'")
        self.values = vals

    # endregion init

    # region method

    def matches(self, keys: Iterable[WindowKey]) -> bool:
        if self.key is None:
            return False

        k = self.key
        for key in keys:
            if k == key:
                return True

        return False

    # endregion method

    # region override

    def __str__(self):
        return f"Key:{self.key}   Values:{self.values}"

    def __getitem__(self, key: WindowKey) -> Optional[Any]:
        return self.values[key] if key in self.values else None

    # endregion override


class WindowEventHandler(ABC):
    @abstractmethod
    def handle_window_event(self, event: WindowEvent):
        raise NotImplementedError


class Window(ClassInfo, ClassLogging, sg.Window):

    # region attributes

    DEFAULT_INIT: dict[str, Any] = {
        'font': ('Ariel', 12),
        'resizable': True,
    }

    DEFAULT_OPTIONS: dict[str, Any] = {
        'suppress_error_popups': True,
        'suppress_raise_key_errors': False,
        'suppress_key_guessing': True,
        'warn_button_key_duplicates': True,
    }

    PSGWINDOW_ID_2_WINDOW: dict[int, Self] = {}

    # endregion attributes

    # region init

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **(self.__class__.DEFAULT_INIT | kwargs))
        self.__class__.PSGWINDOW_ID_2_WINDOW[id(self)] = self

        self._window_info_size: tuple[int | None, int | None] = (None, None)
        self._window_info_location: tuple[int | None, int | None] = (None, None)
        self._window_size_check_size: tuple[int | None, int | None] = (None, None)
        self._event_handlers: list[Any] = []

        self.key: WindowKey = WindowKey(f"{self._class_name}[{self._class_instance_id}]")
        self.key_config: WindowKey = self.key.child('config')
        self.key_config_size: WindowKey = self.key_config.child('size')
        self.key_config_location: WindowKey = self.key_config.child('location')

    # endregion init

    # region @classmethod

    @classmethod
    def theme_background_color(cls) -> str:
        return sg.theme_background_color()

    @classmethod
    def theme(cls, new_theme: str = None) -> str:
        return sg.theme(new_theme=new_theme)

    @classmethod
    def set_options(cls, *args, **kwargs) -> None:
        sg.set_options(*args, **(cls.DEFAULT_OPTIONS | kwargs))

    # endregion @classmethod

    # region @property

    @property
    def size(self) -> tuple[int | None, int | None]:
        size = self.current_size_accurate()
        if size[0] is None and size[1] is None:
            return self.size
        return size

    @property
    def location(self) -> tuple[int | None, int | None]:
        loc = self.current_location(more_accurate=True)
        if loc[0] is None and loc[1] is None:
            return self.current_location(more_accurate=False)
        return loc

    # endregion @property

    # region method

    def run(self):
        self.Finalize()
        self._event_handlers.extend([x for x in self.AllKeysDict.values() if isinstance(x, WindowEventHandler)])
        self._log.debug(f"Found {len(self._event_handlers)} event handlers")
        self.BringToFront()
        self.bind('<Configure>', self.key_config)

        self._window_info_refresh()
        self._check_window_size()

        continue_loop: bool = True
        while continue_loop:
            continue_loop = self._run_loop()

    def _run_loop(self) -> bool:
        window_events: list[WindowEvent] = []

        event, values = self.read()
        window_event = WindowEvent(self, event, values)
        self._log.debug(window_event)

        if window_event.is_exit:
            self._log.debug('Exiting...')
            return False

        window_events.append(window_event)

        self._check_window_size()
        window_info_changes = self._window_info_refresh()
        window_events.extend([WindowEvent(self, x[0], values) for x in window_info_changes])

        for window_event in window_events:
            for i, event_handler in enumerate(self._event_handlers):
                self._log.debug(f"Calling self._event_handlers[{i}]={event_handler} for event {window_event}")
                event_handler.handle_window_event(window_event)

        self._check_window_size()

        return True

    def _check_window_size(self):
        if self.Resizable:
            return

        size_old_raw = self._window_size_check_size
        size_old = (coalesce(size_old_raw[0], -1), coalesce(size_old_raw[1], -1))
        size_new_raw = self.size
        size_new = (coalesce(size_new_raw[0], -1), coalesce(size_new_raw[1], -1))

        if size_new[0] > size_old[0] or size_new[1] > size_old[1]:
            self._window_size_check_size = size_new_raw
            self._log.debug(f"Locking window size to {size_new[0]} x {size_new[1]}  screensize={sg.Window.get_screen_size()}")
            self.set_min_size(size_new)

    def _window_info_refresh(self) -> list[tuple[WindowKey, Any | None, Any | None]]:
        events: list[tuple[WindowKey, Any | None, Any | None]] = []

        size_old = self._window_info_size
        size_new = self.size
        if size_old != size_new:
            events.append((self.key_config_size, size_old, size_new))
        self._window_info_size = size_new

        location_old = self._window_info_location
        location_new = self.location
        if location_old != location_new:
            events.append((self.key_config_location, location_old, location_new))
        self._window_info_location = location_new

        return events

    # endregion method

    # region override

    def __getitem__(self, item):
        try:
            if item is None:
                raise ValueError("Argument key cannot be None")
            if not isinstance(item, WindowKey):
                item = WindowKey(item)
            element = self.AllKeysDict.get(item)
            if element is not None:
                return element

            element = self.AllKeysDict.get(str(item))
            if element is not None:
                return element

            all_keys = [k for k in self.AllKeysDict.keys()]
            all_keys.sort(key=lambda k: type(k).__name__.casefold())
            all_keys_str_list = [f"{k}  ({type(k).__name__})" for k in all_keys]
            all_keys_str = "\n".join(all_keys_str_list)
            raise KeyError(f"Window does not have element '{item}  ({type(item).__name__})' listing all keys:\n{all_keys_str}")

        except KeyError as e:
            self._log.exception("No key found", exc_info=e)
            raise

    # endregion override


# region Elements


class ElementBase(ClassInfo, ClassLogging, WindowEventHandler):
    """
    Mixin for use by other elements. This must be inherited FIRST to work correctly.
    https://stackoverflow.com/a/50465583
    """

    # region init

    def __init__(self, *args, **kwargs):
        key = kwargs.get('key')
        if key is None:
            raise ValueError("argument 'key' is required")

        if not isinstance(key, WindowKey):
            raise ValueError(f"argument 'key' must be {WindowKey.__name__} but was instead {type(key).__name__} '{key}'")

        if not isinstance(self, sg.Element):
            raise ValueError(f"class {type(self).__name__} must inherit from {type(sg.Element).__name__}")

        super().__init__(*args, **kwargs)  # forwards all unused arguments

    # endregion init

    # region @property

    @property
    def window(self) -> Window:
        pf = getattr(self, 'ParentForm', None)
        if pf is not None and isinstance(pf, Window):
            return pf

        pw = getattr(self, 'ParentWindow', None)
        if pw is not None and isinstance(pw, Window):
            return pw

        if pf is not None and isinstance(pf, sg.Window):
            w = Window.PSGWINDOW_ID_2_WINDOW.get(id(pf))
            if w is not None:
                return w

        if pw is not None and isinstance(pw, sg.Window):
            w = Window.PSGWINDOW_ID_2_WINDOW.get(id(pw))
            if w is not None:
                return w

        if len(Window.PSGWINDOW_ID_2_WINDOW.values()) == 1:
            # if we only have one window, it is probably the one we are looking for
            return first(Window.PSGWINDOW_ID_2_WINDOW.values())

        pf_str = 'None' if pf is None else f"{type(pf).__name__}"
        pw_str = 'None' if pw is None else f"{type(pw).__name__}"

        raise ValueError(
            '   '.join(
                [
                    f"Could not retrieve Window.",
                    f"ParentForm={pf_str}",
                    f"ParentWindow={pw_str}",
                    f"len(PSGWINDOW_ID_2_WINDOW)={len(Window.PSGWINDOW_ID_2_WINDOW.values())}",
                ]
            )
        )

    # endregion @property

    # region override

    def handle_window_event(self, event: WindowEvent):
        raise NotImplementedError

    # endregion override


class ColumnCollapsable(ElementBase, sg.Column):

    # region init

    def __init__(
        self,
        key: WindowKey,
        layout: list[list[sg.Element]],
        title_text: str = None,
        arrow_up: str = sg.SYMBOL_UP,
        arrow_down: str = sg.SYMBOL_DOWN,
        collapsed: bool = True,
        *args,
        **kwargs
    ):
        self.arrow_up = arrow_up
        self.arrow_down = arrow_down

        self.element_arrow = sg.Text(
            text=(arrow_up if collapsed else arrow_down),
            enable_events=True,
            key=key.child('arrow'),
        )

        self.element_title = sg.Text(
            text=coalesce(title_text, key.name),
            enable_events=True,
            key=key.child('title'),
            expand_x=True,
        )

        self.element_section = sg.Column(
            layout=layout,
            key=key.child('section'),
            visible=not collapsed,
            # visible=True,
            expand_x=True,
            # size=(1000, None),
            background_color=WindowColor.YELLOW_LIGHT,
            # justification=sg.TEXT_LOCATION_CENTER,
            # metadata=(self.arrow_down, self.arrow_up),
        )

        column_layout = [[self.element_arrow, self.element_title], [sg.pin(self.element_section)]]

        kwargs['key'] = key
        if 'expand_x' not in kwargs:
            kwargs['expand_x'] = True

        super().__init__(layout=column_layout, *args, **kwargs)

    # endregion init

    # region override

    def handle_window_event(self, event: WindowEvent):
        if event.matches([self.element_arrow.key, self.element_title.key, self.element_section.key]):
            w = self.window
            section = w[self.element_section.key]
            is_visible = section.visible
            section.update(visible=not section.visible)
            arrow_element = w[self.element_arrow.key]
            arrow_element.update(self.arrow_up if is_visible else self.arrow_down)

            if hasattr(section, 'TKColFrame') and hasattr(section.TKColFrame, 'canvas'):
                section.contents_changed()

            column = w[self.key]
            if hasattr(column, 'TKColFrame') and hasattr(column.TKColFrame, 'canvas'):
                column.contents_changed()

    # endregion override


class ButtonWindowEvent(ElementBase, sg.Button):

    # region init

    def __init__(
        self,
        key: WindowKey,
        on_window_event: Callable[[WindowEvent], None],
        *args,
        **kwargs
    ):
        self.on_window_event = on_window_event
        kwargs['key'] = key

        super().__init__(*args, **kwargs)

    # endregion init

    # region override

    def handle_window_event(self, event: WindowEvent):
        self.on_window_event(event)

    # endregion override


class ColumnBrowseDir(ElementBase, sg.Column):

    # region init

    def __init__(
        self,
        key: WindowKey,
        label_text: str = 'Directory',
        show_recursive_checkbox: bool = False,
        show_resolve_button: bool = False,
        default_directory: str | Path | None = None,
        default_recursive_checked: bool = False,
        input_background_color_valid: str | None = WindowColor.GREEN_LIGHT,
        input_background_color_invalid: str | None = WindowColor.RED_LIGHT,
        *args,
        **kwargs
    ):
        self.show_recursive_checkbox = show_recursive_checkbox
        self.show_resolve_button = show_resolve_button
        self.input_background_color_valid = input_background_color_valid
        self.input_background_color_invalid = input_background_color_invalid

        default_directory_str = self._parse_path_resolve_str(default_directory)
        default_directory_str = str(default_directory or '')

        self.element_label = sg.Text(
            label_text,
            key=key.child('label'),
            size=(8, 1),
            justification='right',
        )

        self.element_input = sg.Input(
            key=key.child('input'),
            enable_events=True,
            text_color=WindowColor.BLACK,
            background_color=self._parse_input_background_color(default_directory_str),
            default_text='' if default_directory_str is None else default_directory_str,
        )

        self.element_resolve = sg.Button(
            key=key.child('resolve'),
            button_text='Resolve',
            visible=self.show_resolve_button
        )

        self.element_browse = sg.FolderBrowse(
            button_text='Browse',
            key=key.child('browse'),
            target=str(self.element_input.key),  # https://github.com/PySimpleGUI/PySimpleGUI/issues/6260
            initial_folder=default_directory_str,
        )

        self.element_recursive = sg.Checkbox(
            text='Recursive',
            key=key.child('recursive'),
            visible=self.show_recursive_checkbox,
            default=default_recursive_checked,
        )

        column_layout = [[self.element_label, self.element_input, self.element_resolve, self.element_browse, self.element_recursive]]

        kwargs['key'] = key
        if 'expand_x' not in kwargs:
            kwargs['expand_x'] = True

        super().__init__(layout=column_layout, *args, **kwargs)

    # endregion init

    # region @classmethod

    @classmethod
    def _parse_path(cls, path: str | None) -> Path | None:
        if trim(path) is None:
            return None

        p = None

        try:
            p = Path(path)
        except Exception:
            return None

        try:
            return p.expanduser().resolve()
        except Exception:
            pass

        return p

    @classmethod
    def _parse_is_valid(cls, path: str | None) -> bool | None:
        if trim(path) is None:
            return None

        p = cls._parse_path(path)
        if p is None:
            return False

        return p.exists() and p.is_dir()

    @classmethod
    def _parse_path_resolve_str(cls, path: str | Path | None) -> str | None:
        if path is None:
            return None

        if isinstance(path, Path):
            path = str(path)

        p = cls._parse_path(path)
        if p is None:
            return path

        return str(p)

    # endregion @classmethod

    # region method

    def _parse_input_background_color(self, path: str | None) -> str:
        is_valid = self._parse_is_valid(path)
        if is_valid is None:
            return sg.theme_text_element_background_color()
        elif is_valid:
            return self.input_background_color_valid
        else:
            return self.input_background_color_invalid

    # endregion method

    # region @property

    @property
    def value_dir_str(self) -> str:
        return self.window[self.element_input.key].get()

    @value_dir_str.setter
    def value_dir_str(self, value):
        if value is None:
            value = ''
        else:
            value = str(value)
        self.window[self.element_input.key].update(value=value)

    @property
    def value_dir(self) -> Path | None:
        return self.__class__._parse_path(self.value_dir_str)

    @property
    def value_is_valid(self) -> bool:
        return self.__class__._parse_is_valid(self.value_dir_str) or False

    @property
    def value_is_recursive(self) -> bool:
        if not self.show_recursive_checkbox:
            return False
        w = self.window
        recursive = w[self.element_recursive.key]
        return recursive.get()

    # endregion @property

    # region override

    def handle_window_event(self, event: WindowEvent):
        w = self.window

        # update recursive checkbox visibility if changed
        recursive = w[self.element_recursive.key]
        if self.show_recursive_checkbox != recursive.visible:
            recursive.update(visible=self.show_recursive_checkbox)

        # update resolve button visibility if changed
        resolve = w[self.element_resolve.key]
        if self.show_resolve_button != resolve.visible:
            resolve.update(visible=self.show_resolve_button)

        if event.matches([self.element_input.key]):
            c = self._parse_input_background_color(self.value_dir_str)
            w[self.element_input.key].update(background_color=c)

        if event.matches([self.element_resolve.key]):
            p = self.value_dir_str
            pp = self._parse_path_resolve_str(p)
            self._log.debug(f"Resolving Path: {p}  to  {pp}")
            self.value_dir_str = pp

    # endregion override


class TreeRow:

    # region init

    def __init__(
        self,
        parent: str | int | tuple | Self | object | None = None,
        key: str | int | tuple | object | None = None,
        icon: bytes | None = None,
        *args: Any,
    ):
        self.parent = parent
        self.key = key
        self.icon = icon
        self.values: list[Any] = [] if args is None else [x for x in args]

    # endregion init

    # region method

    def insert(self, tree_data: sg.TreeData):
        parent = self.parent
        if parent is None:
            parent = ''
        elif isinstance(parent, TreeRow):
            parent = parent.key
            if parent is None:
                parent = ''

        key = self.key
        if key is None:
            raise ValueError('key is required')

        if len(self.values) == 0:
            raise ValueError('at least one value is required')
        text = xstr(self.values[0])
        values = [coalesce(x, '') for x in self.values[1:]]
        icon = self.icon
        tree_data.Insert(
            parent=parent,
            key=key,
            text=text,
            values=values,
            icon=icon,
        )

    # endregion method

    # region override

    def __str__(self):
        return f"{self.__class__.__name__}(parent={self.parent}, key={self.key}, values={self.values}, icon=len({len(self.icon)}))"

    # endregion override


class Tree(ElementBase, sg.Tree):

    # region init

    def __init__(
        self,
        key: str | object,
        column_names: Iterable[str],
        rows: list[TreeRow] | None = None,
        *args,
        **kwargs
    ):
        column_names_list = [x for x in column_names]
        self._column_names = tuple(column_names_list)
        self._column_visibility: dict[int, bool] = {c: True for c in range(len(column_names_list))}
        self.rows: list[TreeRow] = [] if rows is None else rows

        visible_column_map = kwargs.get('visible_column_map')
        if visible_column_map is not None:
            for i, b in enumerate(visible_column_map):
                if not isinstance(b, bool):
                    raise TypeError(f"Expecting visible_column_map[{i}] to be a bool, but got {type(b).__name__}  {b}")
                if i not in self._column_visibility:
                    raise ValueError(f"Expecting len(visible_column_map)={i} to be less than {len(visible_column_map)}")
                self._column_visibility[i] = b

        self._columns_name_2_index: dict[str, int] = {c: i for i, c in enumerate(column_names_list)}
        self._columns_name_casefold_2_index: dict[str, int] = {c.casefold(): i for i, c in enumerate(column_names_list)}

        if 'col0_heading' in kwargs:
            raise ValueError("'col0_heading' should not be provided as an argument, use 'column_names' instead")
        kwargs['col0_heading'] = column_names_list[0]

        if 'headings' in kwargs:
            raise ValueError("'headings' should not be provided as an argument, use 'column_names' instead")
        kwargs['headings'] = column_names_list[1:]

        kwargs['key'] = key

        if 'data' in kwargs:
            raise ValueError("'data' should not be provided as an argument, use 'rows' instead")
        td = sg.TreeData()
        for row in self.rows:
            row.insert(td)
        kwargs['data'] = td

        if 'max_col_width' not in kwargs:
            kwargs['max_col_width'] = 9999

        super().__init__(*args, **kwargs)

    # endregion init

    # region method

    def _get_column_index(self, name: str) -> int | None:
        if name == '#0':
            return 0

        i = self._columns_name_2_index.get(name)
        if i is None:
            i = self._columns_name_2_index.get(name)
        return i

    def data_refresh(self):
        td = sg.TreeData()
        for row in self.rows:
            row.insert(td)
        self.window[self.key].update(values=td)

    def set_column_sizes(self, column_sizes: dict[str | int, int]) -> None:
        column_sizes_int: dict[int, int | bool] = self._column_visibility.copy()
        for key, value in column_sizes.items():
            if isinstance(key, str):
                col_index = self._get_column_index(key)
                if col_index is None:
                    raise KeyError(f"No column named '{key}'")
            elif isinstance(key, int):
                col_index = key
            else:
                raise KeyError(f"Column must be either int or str but was ({type(key).__name__})  {key}")
            if col_index not in column_sizes_int:
                raise KeyError(f"No column at index {col_index}    (0-{len(self.column_names)})")
            column_sizes_int[col_index] = value

        column_visibility_new: dict[int, bool] = {}
        displayed_columns = []
        column_sizes_new: list[tuple[int, str, int]] = []

        for i, v in sorted(column_sizes_int.items()):
            col_name = '#0' if i == 0 else self._column_names[i]
            if isinstance(v, bool):
                if v:
                    column_visibility_new[i] = True
                    displayed_columns.append(col_name)
                else:
                    column_visibility_new[i] = False
            elif isinstance(v, int):
                column_sizes_new.append((i, col_name, v))
                if v > 0:
                    column_visibility_new[i] = True
                    displayed_columns.append(col_name)
                else:
                    column_visibility_new[i] = False
            else:
                raise ValueError(f"Column type {type(v).__name__} with value {v} for index {i} was not expected")

        widget = self.window[self.key].Widget
        widget.configure(displaycolumns=displayed_columns)
        self._column_visibility = column_visibility_new

        self_name = f"{self.__class__.__name__}[{self.key}]"
        for col_name, i, col_size in column_sizes_new:
            # _log.debug(f"{self_name}.Widget.column[{i}]({col_name}, width={col_size})")
            widget.column(col_name, width=col_size)

        # _log.debug(f"{self_name}.Widget.pack(side='left', fill='both', expand=True)")
        widget.pack(side='left', fill='both', expand=True)

    # endregion method

    # region @property

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._column_names

    # endregion @property

    # region override

    def handle_window_event(self, event: WindowEvent):
        pass

    # endregion override


# endregion Elements


def window_element_theme_sample_list(
    window: Window,
    key: WindowKey,
    label_text: str = 'Themes',
) -> list[sg.Element]:
    label = sg.Text(
        label_text,
        key=key.child('label'),
        size=(8, 1),
        justification='right',
    )
    _log.debug(f"window.background_color: {window.background_color}")

    combo_items = sg.theme_list()
    combo = sg.DropDown(
        values=combo_items,
        enable_events=True,
        key=key.child('combo'),
        readonly=True,
    )

    layout = [label, combo]

    def event_handler(event: WindowEvent):
        val = trim(event[combo.key])
        if val is not None:
            sg.theme(val)
            sg.popup_get_text(f"This is {val}")

    window.subscribe([combo.key], event_handler)

    return layout


def _resize_table_calc(total_width: int | None, column_sizes: [int | float | None]) -> [int]:
    if total_width is not None and total_width < 0:
        raise ValueError(f"total_width={total_width} must be positive")

    if column_sizes is None:
        raise ValueError(f"column_sizes must not be None")

    if len(column_sizes) == 0:
        raise ValueError(f"column_sizes must contain at least one column")

    cols_fixed: dict[int, int] = {}
    cols_dyn: dict[int, float | None] = {}

    for i, column_size in enumerate(column_sizes):
        if column_size is None:
            cols_dyn[i] = column_size
        elif isinstance(column_size, float):
            if column_size <= 0.0:
                raise ValueError(f"column_sizes[{i}] '{column_size}' must be greater than 0.0")
            cols_dyn[i] = column_size
        elif isinstance(column_size, int):
            if column_size < 0:
                raise ValueError(f"column_sizes[{i}] '{column_size}' must be greater than 0")
            cols_fixed[i] = column_size
        else:
            raise ValueError(f"column_sizes[{i}] '{column_size}' is not an int or float")

    assert len(cols_fixed) + len(cols_dyn) == len(column_sizes)

    cols_fixed_sum = sum(cols_fixed.values()) if len(cols_fixed) > 0 else 0

    if total_width is not None and cols_fixed_sum > total_width:
        raise ValueError(f"cols_fixed_sum={cols_fixed_sum} cannot be greater than total_width={total_width}")

    if len(cols_fixed) == len(column_sizes):
        assert len(cols_dyn) == 0
        assert cols_fixed_sum == sum(column_sizes)
        return column_sizes.copy()

    assert len(cols_dyn) > 0
    if total_width is None:
        raise ValueError(f"total_width cannot be None when dynamic columns are provided")
    assert total_width is not None

    assert cols_fixed_sum <= total_width

    num_columns = len(column_sizes)
    all_column_sizes_calculated: [int] = []

    for i in range(num_columns):
        px = cols_fixed.get(i)
        all_column_sizes_calculated.append(0 if px is None else px)

    total_width_percent_defined: float = 0.0
    for f in cols_dyn.values():
        if f is not None:
            total_width_percent_defined += f

    if total_width_percent_defined > 1.0:
        raise ValueError(f"total_width_percent_defined={total_width_percent_defined} cannot be greater than 1.0")

    if None not in cols_dyn.values() and total_width_percent_defined < 1.0:
        raise ValueError(f"total_width_percent_defined={total_width_percent_defined} must total 1.0 if no None dynamic columns are provided")

    total_width_percent_undefined: float = 1.0 - total_width_percent_defined
    if None in cols_dyn.values():
        col_indexes_where_none = [x[0] for x in cols_dyn.items() if x[1] is None]
        assert len(col_indexes_where_none) > 0
        percent_for_none_columns = total_width_percent_undefined / float(len(col_indexes_where_none))
        for col_index in col_indexes_where_none:
            cols_dyn[col_index] = percent_for_none_columns

    assert None not in cols_dyn.values()
    assert round(sum(cols_dyn.values())) == 1

    total_width_remaining: int = total_width - cols_fixed_sum

    for i, percent in cols_dyn.items():
        all_column_sizes_calculated[i] = round(percent * total_width_remaining)

    pixel_adjustment = total_width - sum(all_column_sizes_calculated)
    while pixel_adjustment != 0:
        positive_or_negative: int = 1 if pixel_adjustment > 0 else -1

        total_dynamic_column_px = sum([all_column_sizes_calculated[i] for i in cols_dyn.keys()])
        if total_dynamic_column_px == 0:
            raise ValueError(f"total_dynamic_column_px={total_dynamic_column_px} but pixel_adjustment={pixel_adjustment} so not sure how to fix")

        for i in cols_dyn.keys():
            if all_column_sizes_calculated[i] > 0:
                all_column_sizes_calculated[i] += (1 * positive_or_negative)
            pixel_adjustment = total_width - sum(all_column_sizes_calculated)
            if pixel_adjustment == 0:
                break

    assert total_width == sum(all_column_sizes_calculated)

    return all_column_sizes_calculated


def resize_table(table: sg.Element, total_width: int | None, column_sizes: [int | float | None]) -> None:
    """
    Resizes columns of table or tree to specified sizes.

    :rtype: None
    :param table: The Table or Tree to resize
    :param total_width:
    :param column_sizes:

    Example column_sizes:
        1) [20, 30, 40] resizes the columns of table as...
            - Column0: 20px
            - Column1: 30px
            - Column2: 40px
            - and ignores the total_width

        2) [None, 30, 40] with a total_width of 200px...
            - Column0: 130px (100%)
            - Column1: 30px
            - Column2: 40px

        3) [None, None, 40] with a total_width of 200px...
            - Column0: 80px (50%)
            - Column1: 80px (50%)
            - Column2: 40px

        4) [0.2, 0.8, 40] with a total_width of 200px...
            - Column0: 32px  (20%)
            - Column1: 128px (80%)
            - Column2: 40px

        5) [None, 0.2, 0.4, 40] with a total_width of 200px...
            - Column0: 64px (40%)
            - Column1: 32px (20%)
            - Column2: 64px (40%)
            - Column3: 40px

    """
    table_name = f"{table.__class__.__name__}[{table.key}]"

    # from tkinter import ttk
    table_widget = table.Widget
    table_name_widget = f"{table_name}.Widget"

    # _log.debug('table attributes:\n' + tostring_attributes(table, multiline=True))

    column_names = []
    if hasattr(table, 'col0_heading'):
        column_names.append(table.col0_heading)
    if hasattr(table, 'ColumnHeadings'):
        column_headings = table.ColumnHeadings
        if column_headings is not None:
            column_names.extend(column_headings)

    grid_size = table_widget.grid_size()
    _log.debug(f"{table_name_widget}.grid_size() = {grid_size}")
    num_columns = grid_size[0]

    if num_columns == 0:
        num_columns = len(column_names)

    if len(column_sizes) != num_columns:
        raise ValueError(f"column_sizes={len(column_sizes)} must equal num_columns={num_columns}")

    new_column_sizes = _resize_table_calc(total_width=total_width, column_sizes=column_sizes)
    assert num_columns == len(new_column_sizes)

    _log.debug(f"{table_name}.expand(expand_x=True, expand_y=True)")
    table.expand(expand_x=True, expand_y=True)

    _log.debug(f"{table_name_widget}.pack_forget()")
    table_widget.pack_forget()

    displayed_columns = []
    for i, column_name in enumerate(column_names):
        if i == 0:
            column_name = '#0'
        new_column_size = new_column_sizes[i]
        if new_column_size > 0:
            displayed_columns.append(column_name)

    table_widget.configure(displaycolumns=displayed_columns)

    for i, column_name in enumerate(column_names):
        if i == 0:
            column_name = '#0'
        new_column_size = new_column_sizes[i]
        _log.debug(f"{table_name_widget}.column({column_name}, width={new_column_size})")
        table_widget.column(column_name, width=new_column_size)

    _log.debug(f"{table_name_widget}.pack(side='left', fill='both', expand=True)")
    table_widget.pack(side='left', fill='both', expand=True)
