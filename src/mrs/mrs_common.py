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

import base64
import colorsys
import hashlib
import inspect
import json as jsn
import logging
import os
import re
import sys
import threading
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from enum import auto, Enum, IntFlag, StrEnum
from inspect import FrameInfo
from logging import Logger, LoggerAdapter
from os import DirEntry, PathLike, stat_result
from pathlib import Path, PurePath
from types import FrameType, ModuleType
from typing import *
from typing import Any
from uuid import UUID

_log = logging.getLogger('mrs_common')

T = TypeVar('T')


# region util


def trim(s: str | Iterable[str] | None, exclude_none=True) -> str | List[str | None] | List[str] | None:
    if s is None:
        return None
    if isinstance(s, str):
        s = s.strip()
        return None if len(s) == 0 else s

    # iterable
    result: List[str | None] = []
    for item in s:
        if item is None:
            if not exclude_none:
                result.append(item)
        else:
            item = item.strip()
            if len(item) == 0:
                if not exclude_none:
                    result.append(None)
            else:
                result.append(item)
    return result


def trim_casefold(s: str | None) -> str | None:
    return s if s is None else trim(s.casefold())


def coalesce(*args: Any) -> Any:
    if args is None:
        return None
    for item in args:
        if item is not None:
            return item
    return None


_BOOL_TRUE: Set[str] = {'true', '1', 't', 'y', 'yes'}
_BOOL_FALSE: Set[str] = {'false', '0', 'f', 'n', 'no'}


def bool_parse(value: str | bytes | bool) -> bool:
    if value is None:
        raise TypeError("bool_parse() argument must be a string or a bytes-like object, not 'NoneType'")
    v = trim(str(value))
    if v is not None:
        v = v.lower()
        if v in _BOOL_TRUE:
            return True
        if v in _BOOL_FALSE:
            return False
    raise ValueError("invalid literal for bool_parse(): " + value.__repr__())


def split(delimiters: str | Iterable[str], string: str, maxsplit: int = 0, remove_empty: bool = False) -> list[str]:
    """
    Split a string on multiple characters
    https://stackoverflow.com/a/13184791

    :param delimiters: the delimiters to split the string on
    :param string: the string to actually split
    :param maxsplit: the number of times to split the string. 0 means no limit
    :param remove_empty: whether to remove empty strings
    :return: a list of strings split
    """
    if isinstance(delimiters, str):
        delimiters = [delimiters, ]

    regex_pattern = '|'.join(map(re.escape, delimiters))
    lst = re.split(regex_pattern, string, maxsplit)
    if remove_empty:
        lst2 = []
        for s in lst:
            if len(s) > 0:
                lst2.append(s)
        lst = lst2
    # print_error(f"Split string to {len(lst)} parts:  {lst}")
    return lst


def split_on_capital(string: str, maxsplit: int = 0) -> list[str]:
    """
    Split a string on capitalized characters
    https://stackoverflow.com/a/2279177
    :param string: the string to actually split
    :param maxsplit: the number of times to split the string. 0 means no limit
    :return: a list of strings split
    """
    return re.sub(r"([A-Z])", r" \1", string, maxsplit).split()


def url_join(*args: Any | None) -> str | None:
    """
    https://stackoverflow.com/a/11326230
    """
    if args is None:
        return None
    args_new = ""
    for arg in args:
        arg = trim(xstr(arg))
        if arg is None:
            continue
        arg = trim(arg.rstrip('/'))
        if arg is None:
            continue
        if len(args_new) > 0:
            args_new += "/"
        args_new += arg
    return trim(args_new)


def first(iterable: Iterable[Any]) -> Any:
    next((x for x in iterable), None)


def xstr(s):
    if s is None:
        return ''
    elif isinstance(s, str):
        return s
    elif isinstance(s, Iterable):
        return [xstr(x) for x in s]
    else:
        return str(s)


def str2base64(s: str | None, encoding: str = "utf-8"):
    if s is None:
        return None
    return base64.b64encode(s.encode(encoding)).decode(encoding)


def json2base64(json: str | Any | None, indent: int | None = None, encoding: str = "utf-8") -> str | None:
    return str2base64(json2str(json, indent=indent), encoding=encoding)


def json2str(json: str | Any | None, indent: int | None = None) -> str | None:
    if json is None:
        return None
    if isinstance(json, str):  # already json
        json = json2obj(json)
    return jsn.dumps(json, indent=indent)


def json2obj(json: str | None) -> Any | None:
    if json is None:
        return None
    assert isinstance(json, str)
    return jsn.loads(json)


def datetime_now():
    return datetime.now(timezone.utc).astimezone()


def trim_dict(
    items: Mapping[str, str | None] | [Tuple[str | None, str | None]] | [[str | None]],
    exclude_none_values=False
) -> Dict[str, str | None]:
    if isinstance(items, Mapping):
        items = [(k, v) for k, v in items.items()]

    result: Dict[str, str | None] = {}
    for item in items:
        k = trim(item[0])
        if k is None:
            continue
        v = trim(item[0])
        if v is None and exclude_none_values:
            continue
        result[k] = v

    return result


def print_error(*args, **kwargs):
    """
    https://stackoverflow.com/a/14981125
    """
    print(*args, file=sys.stderr, **kwargs)


def tostring_attributes(obj, included=None, excluded=None, use_repr=False, multiline=False) -> str:
    result = ""
    for name in sorted(obj.__dir__(), key=lambda x: x.casefold()):
        if included is not None and name not in included:
            continue
        if excluded is not None and name in excluded:
            continue
        if name.startswith("_"):
            continue
        if name.startswith("__"):
            continue
        if not hasattr(obj, name):
            continue
        v = getattr(obj, name)
        vt = type(v)
        # print(f"Found attribute '{name}'={v}  ({vt.__name__})")
        if vt.__name__ == 'method':
            continue

        if v is None:
            v = "None"
        elif isinstance(v, str):
            v = "'" + v + "'"
        elif use_repr:
            v = repr(v)
        elif not use_repr:
            v = str(v)

        if len(result) > 0:
            if multiline:
                result += f"\n"
            else:
                result += ", "
        if multiline:
            result += '  '
        result += f"{name}={v}"
    return result


def decimal_round(d: Decimal, places: int):
    # https://stackoverflow.com/a/8869027
    places *= -1
    return d.quantize(Decimal(10) ** places).normalize()


def str_convert_to_type(value: Any | None, type_to_convert_to: type, trim_if_string_result: bool = False) -> Any | None:
    if value is None:
        return None

    if type_to_convert_to == str:
        value_str = value if isinstance(value, str) else str(value)
        if trim_if_string_result:
            value_str = trim(value_str)
        return value_str

    try:
        value_str = trim(str(value))
        if value_str is None:
            return None
        elif type_to_convert_to == bool:
            return bool_parse(value_str)
        elif type_to_convert_to == int:
            return int(value_str)
        elif type_to_convert_to == float:
            return float(value_str)
        else:
            raise NotImplementedError(
                f"No handler available to convert '{value}' to type {type_to_convert_to.__name__}"
            )
    except ValueError as ve:
        raise ValueError(f"Could not convert '{value}' to type {type_to_convert_to.__name__}") from ve


def iterable_flatten(
    item: Any,
    parser: Callable[[Any], Any] | None = None,
) -> list[Any]:
    """
    Loops through all arguments and adds them to a return list. If any of the arguments are
    an Iterable then loops over the Iterable and adds them to the return list, and so on, and so on.

    Note: possible 'infinite recursion' if `parser` returns multiple items that are then fed back into
    parser, which then returns more and more items, etc
    """
    return_items: list[Any] = []
    stack: list[Any] = [item, ]
    while stack:
        # if len(stack) > 20 or len(return_items) > 20:
        #    raise ValueError(f"Too many items\nstack={stack}\n\nreturn_items={return_items}\n")
        current = stack.pop()

        if parser is not None:
            current = parser(current)

        if current is None:
            return_items.append(current)
        elif isinstance(current, str):
            return_items.append(current)
        elif isinstance(current, Iterable):
            if isinstance(current, Reversible):
                stack.extend(reversed(current))
            else:
                stack.extend(reversed(list(current)))
        else:
            return_items.append(current)

    return return_items


class Object:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# noinspection PyBroadException
def stack_currentframe() -> Optional[FrameType]:
    frame_sys = None
    frame_traceback = None
    frame_inspect = None

    try:
        if hasattr(sys, '_getframe'):
            # noinspection PyProtectedMember
            frame_sys = sys._getframe()
            if frame_sys is not None:
                frame_back = frame_sys.f_back
                if frame_back is not None:
                    return frame_back
    except Exception:
        pass

    try:
        raise Exception
    except Exception:
        try:
            frame_traceback = sys.exc_info()[2].tb_frame
            if frame_traceback is not None:
                frame_back = frame_traceback.f_back
                if frame_back is not None:
                    return frame_back
        except Exception:
            pass

    try:
        frame_inspect = inspect.currentframe()
        if frame_inspect is not None:
            frame_back = frame_traceback.f_back
            if frame_back is not None:
                return frame_back
    except Exception:
        pass

    # Just return the current frame if we have it
    if frame_inspect is not None:
        return frame_inspect
    if frame_traceback is not None:
        return frame_traceback
    if frame_sys is not None:
        return frame_sys

    # or ANY frame
    frame_stack = inspect.stack()
    for f in frame_stack:
        if f is not None and f.frame is not None:
            return f.frame

    return None


_stack_current_file = os.path.normcase(trim.__code__.co_filename)


def _stack_get_first_frame_not_in_this_file() -> Optional[FrameType]:
    def is_internal_frame(_frame):
        if _frame is None:
            return False
        filename = os.path.normcase(_frame.f_code.co_filename)
        return filename == _stack_current_file or ('importlib' in filename and '_bootstrap' in filename)

    frame = stack_currentframe()
    while is_internal_frame(frame):
        frame_next = frame.f_back
        if frame_next == frame:
            return None  # prevent infinite loop
        frame = frame_next
    return frame


def parse_module_name(module: Any):
    def isvalid(s: str):
        return True if s is not None and s != '__main__' else False

    name: Optional[str]
    match module:
        case None:
            name = None
        case str():
            name = module
        case ModuleType():
            name = module.__name__
        case type():
            name = module.__module__
        case _:
            name = module.__module__

    name = trim(name)
    if isvalid(name):
        return name
    module_type = sys.modules.get('__main__')
    if module_type is not None:
        name = trim(module_type.__name__)
        if isvalid(name):
            return name
        d = vars(module_type)
        if d is not None:
            o = d.get('__package__')
            if o is not None:
                name = trim(str(o))
                if isvalid(name):
                    return name

    frame = _stack_get_first_frame_not_in_this_file()
    if frame is not None:
        code = frame.f_code
        if code is not None:
            filename = trim(code.co_filename)
            if filename is not None:
                name = trim(inspect.getmodulename(filename))
                if isvalid(name):
                    return name

    return '__main__'


def parse_class_name(cls: Any) -> Optional[str]:
    name: Optional[str]
    match cls:
        case None:
            name = None
        case str():
            name = cls
        case type():
            name = cls.__name__
        case object():
            name = cls.__class__.__name__
        case _:
            name = cls.__class__.__name__
    return trim(name)


def logger(module: Any = None, cls: Any = None) -> logging.Logger:
    cls_name = parse_class_name(cls)
    cls_name = ('.' + cls_name) if cls_name is not None else ''
    module_name = parse_module_name(module if module is not None else cls)
    return logging.getLogger(module_name + cls_name)


def compare(
    x: Any | None,
    y: Any | None,
) -> int:
    if x is None:
        return 0 if y is None else -1
    if y is None:
        return 1
    if x is y:
        return 0

    es = {}
    i = 0
    if hasattr(x, '__lt__'):
        try:
            if x < y:
                return -1
            i = i + 1
            if i == 2:
                return 0
        except Exception as e:
            es[0] = e

    if hasattr(x, '__gt__'):
        try:
            if x > y:
                return 1
            i = i + 1
            if i == 2:
                return 0
        except Exception as e:
            es[1] = e

    if hasattr(y, '__lt__'):
        try:
            if y < x:
                return 1
            i = i + 1
            if i == 2:
                return 0
        except Exception as e:
            es[2] = e

    if hasattr(y, '__gt__'):
        try:
            if y > x:
                return -1
            i = i + 1
            if i == 2:
                return 0
        except Exception as e:
            es[3] = e

    if len(es) > 0:
        raise ExceptionGroup(f"Exceptions encountered comparing objects {x.__class__.__name__} and {y.__class__.__name__}", [e for e in es.values()])

    raise TypeError(f"Objects {x.__class__.__name__} and {y.__class__.__name__} are not comparable")


def compare_iterable(
    x: Iterable[Any] | None,
    y: Iterable[Any] | None,
    comparator: Callable[[Any, Any], int] = None,
) -> int:
    if isinstance(x, str):
        raise TypeError(f"x='{x}' is a str not a real iterable")
    if isinstance(y, str):
        raise TypeError(f"y='{y}' is a str not a real iterable")

    if comparator is None:
        comparator = compare

    if x is None:
        return 0 if y is None else -1
    if y is None:
        return 1

    x = iter(x)
    y = iter(y)

    stop = Object()
    while True:
        xx = next(x, stop)
        yy = next(y, stop)

        if xx is stop:
            return 0 if yy is stop else -1
        if yy is stop:
            return 1

        c = comparator(xx, yy)
        if c != 0:
            return c


# endregion util


# region dict


class DictStr(MutableMapping):
    def __init__(self, data=None, key_converter: Callable[[Any], Any] = None, **kwargs):
        if key_converter is None:
            key_converter = lambda x: x
        self.key_converter = key_converter
        self._data = dict()
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def _convert_key(self, s: str) -> str:
        return self.key_converter(s)

    def __setitem__(self, key, value):
        self._data[self._convert_key(key)] = (key, value)

    def __getitem__(self, key):
        return self._data[self._convert_key(key)][1]

    def __delitem__(self, key):
        del self._data[self._convert_key(key)]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in self._data.values())

    def __len__(self):
        return len(self._data)

    def items_strfunc(self):
        return ((casedkey, keyval[1]) for (casedkey, keyval) in self._data.items())

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = self.__class__(other)
        else:
            return NotImplemented

        # Compare insensitively
        return dict(self.items_strfunc()) == dict(other.items_strfunc())

    def copy(self):
        return self.__class__(self._data.values())

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return f"{dict(self.items())}"


class DictKeyString:
    """
    https://stackoverflow.com/a/30221547
    """

    def __init__(self, key: str, case_sensitive: bool = True):
        super(DictKeyString, self).__init__()
        self.value = key
        self.key = key if case_sensitive else key.casefold()
        self.hash = hash(self.key)

    def __hash__(self): return self.hash

    def __eq__(self, other): return self.key == other.key

    def __str__(self): return self.value


# endregion dict


# region os


class OperatingSystem(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value: int, prefixes: list[str], is_fs_case_sensitive: bool):
        self._value = value
        self._prefixes = prefixes
        self._is_fs_case_sensitive = is_fs_case_sensitive

    def __str__(self):
        return self.name

    @property
    def is_fs_case_sensitive(self) -> bool:
        return self._is_fs_case_sensitive

    UNKNOWN = 1, [], True
    WINDOWS = 2, ['win32', 'win'], False
    MAC = 3, ['darwin'], True
    FREEBSD = 4, ['freebsd'], True
    AIX = 5, ['aix'], True
    LINUX = 6, ['linux'], True
    WINDOWS_CYGWIN = 7, ['cygwin'], True

    @staticmethod
    def current() -> OperatingSystem:
        p = sys.platform.lower()
        for e in OperatingSystem:
            for prefix in e._prefixes:
                if p.startswith(prefix.lower()):
                    return e

        return OperatingSystem.UNKNOWN


# endregion os


# region runtime


class RuntimeExecutionType(Enum):
    SOURCE = 1
    INTERACTIVE = 2
    PYINSTALLER_FOLDER = 3
    PYINSTALLER_FILE = 4

    def __str__(self):
        return self.name


class RuntimeInfo:
    """
    Details about the current runtime environment
    """

    def __init__(self):
        self.project_root_dir = self.__class__.get_project_root_dir()
        self.frozen = getattr(sys, 'frozen', False)
        self.meipass = getattr(sys, '_MEIPASS', None)
        self.sys_executable = sys.executable
        self.file = __file__
        self.cwd = os.getcwd()
        self.os = OperatingSystem.current()
        self.version = sys.version
        self.version_info = sys.version_info
        self.version_hex = sys.hexversion

        if self.frozen:
            application_path = os.path.dirname(sys.executable)
            running_mode = 'Frozen/executable'
            execution_type = RuntimeExecutionType.PYINSTALLER_FOLDER
            from pathlib import Path
            for path_part in Path('C:/path/to/file.txt').parts:
                # https://pyinstaller.org/en/stable/operating-mode.html
                if path_part and path_part.upper().startswith('_MEI') and len(path_part) == len('_MEIxxxxxx'):
                    execution_type = RuntimeExecutionType.PYINSTALLER_FILE
                    break
        else:
            try:
                application_path = os.path.dirname(os.path.realpath(__file__))
                running_mode = "Non-interactive (e.g. 'python myapp.py')"
                execution_type = RuntimeExecutionType.SOURCE
            except NameError:
                application_path = os.getcwd()
                running_mode = 'Interactive'
                execution_type = RuntimeExecutionType.INTERACTIVE

        self.application_path = application_path
        self.running_mode = running_mode
        self.execution_type = execution_type

    @staticmethod
    def get_project_root_dir() -> str:
        """
        Returns the name of the project root directory.
        https://stackoverflow.com/a/62510836
        :return: Project root directory name
        """

        # stack trace history related to the call of this function
        frame_stack: [FrameInfo] = inspect.stack()

        # get info about the module that has invoked this function
        # (index=0 is always this very module, index=1 is fine as long this function is not called by some other
        # function in this module)
        frame_info: FrameInfo = frame_stack[1]

        # if there are multiple calls in the stacktrace of this very module, we have to skip those and take the first
        # one which comes from another module
        if frame_info.filename == __file__:
            for frame in frame_stack:
                if frame.filename != __file__:
                    frame_info = frame
                    break

        # path of the module that has invoked this function
        caller_path: str = frame_info.filename

        # absolute path of the module that has invoked this function
        caller_absolute_path: str = os.path.abspath(caller_path)

        # get the top most directory path which contains the invoker module
        paths: [str] = [p for p in sys.path if p in caller_absolute_path]
        paths.sort(key=lambda p: len(p))
        caller_root_path: str = paths[0]

        if not os.path.isabs(caller_path):
            # file name of the invoker module (eg: "mymodule.py")
            caller_module_name: str = Path(caller_path).name

            # this piece represents a subpath in the project directory
            # (e.g. if the root folder is "myproject" and this function has been called from
            # myproject/foo/bar/mymodule.py this will be "foo/bar")
            project_related_folders: str = caller_path.replace(os.sep + caller_module_name, '')

            # fix root path by removing the undesired subpath
            caller_root_path = caller_root_path.replace(project_related_folders, '')

        p = Path(caller_root_path)
        # return p.name
        return str(p.resolve())

    def log_runtime_info(self, log: Logger, level: int = logging.DEBUG) -> None:
        log.log(level=level, msg='RUNTIME_INFO:')
        for attr_name, attr_value in sorted(vars(self).items(), key=lambda x: x[0].lower()):
            log.log(level=level, msg=f"  {attr_name}: {attr_value}")

    @staticmethod
    def print_old_way_values() -> ():
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
            running_mode = 'Frozen/executable'
        else:
            try:
                app_full_path = os.path.realpath(__file__)
                application_path = os.path.dirname(app_full_path)
                running_mode = "Non-interactive (e.g. 'python myapp.py')"
            except NameError:
                application_path = os.getcwd()
                running_mode = 'Interactive'

        config_full_path = os.path.join(application_path, 'myconfig.cfg')

        print('Running mode:', running_mode)
        print('  Application path:', application_path)
        print('  Config full path:', config_full_path)


RUNTIME_INFO = RuntimeInfo()

# endregion runtime


# region fs


FS_BUFFERSIZE = 1024 * 1024 * 20


def __fs_datetime_parse(time: float) -> datetime:
    return datetime.fromtimestamp(time, tz=timezone.utc)


def fs_accessed(stat: stat_result) -> datetime:
    return __fs_datetime_parse(stat.st_atime)


def fs_modified(stat: stat_result) -> datetime:
    return __fs_datetime_parse(stat.st_mtime)


def fs_created(stat: stat_result) -> datetime:
    if RUNTIME_INFO.os in [OperatingSystem.WINDOWS, OperatingSystem.WINDOWS_CYGWIN]:
        return __fs_datetime_parse(stat.st_ctime)
    else:
        try:
            return __fs_datetime_parse(stat.st_birthtime)
        except AttributeError:
            return __fs_datetime_parse(stat.st_mtime)


def fs_size(stat: stat_result) -> int:
    return stat.st_size


def fs_list(
    path: str, recursive: bool = False, follow_symlinks: bool = False, include_files: bool = True,
    include_dirs: bool = True, include_symlinks: bool = False, sort_items: bool = True
) -> list[DirEntry]:
    """Gets DirEntry objects for given directory."""
    items = []
    if not include_files and not include_dirs:
        return items
    _log.debug(f"fs_list({repr(path)})")

    def add_item(de: DirEntry) -> bool:
        if de is None:
            return False
        try:
            if include_files and de.is_file(follow_symlinks=follow_symlinks):
                items.append(de)
            elif include_dirs and de.is_dir(follow_symlinks=follow_symlinks):
                items.append(de)
            elif include_symlinks and de.is_symlink():
                items.append(de)
            return de.is_dir(follow_symlinks=follow_symlinks)
        except OSError:
            _log.exception(f"Error checking path type for path: {de.path}")
            return False

    if not recursive:
        with os.scandir(path) as it:
            for entry in it:
                add_item(entry)
    else:
        dirs_parsed = set()
        dirs_stack = deque()
        dirs_stack.append(path)
        while dirs_stack:
            dir_current = dirs_stack.pop()
            if dir_current in dirs_parsed:
                continue
            dirs_parsed.add(dir_current)
            try:
                with os.scandir(str(dir_current)) as it:
                    for entry in it:
                        if add_item(entry) and entry.path not in dirs_parsed:
                            dirs_stack.append(entry.path)
            except OSError:
                _log.exception(f"Error os.scandir for path: {dir_current}")

    if sort_items:
        items = sorted(items, key=lambda item: PurePath(item))
    return items


def fs_file_read_chunked(stream: IO[bytes], buffersize: Optional[int] = None) -> Iterator[bytes]:
    """
    Lazy function (generator) to read a file piece by piece.
    https://stackoverflow.com/a/519653
    """
    bs = buffersize if buffersize is not None else FS_BUFFERSIZE
    while True:
        data = stream.read(bs)
        if not data:
            break
        yield data


def fs_file_hash(path: Union[str, bytes, PathLike[str], PathLike[bytes] | int], hasher, result_hex: bool = True) -> \
        Union[bytes, str]:
    # hasher = hashlib.sha1()
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    with open(path, 'rb') as f:
        for d in fs_file_read_chunked(f):
            hasher.update(d)
    return hasher.hexdigest() if result_hex else hasher.digest()


def fs_file_hash_md5(path: Union[str, bytes, PathLike[str], PathLike[bytes] | int]) -> str:
    return fs_file_hash(path, hashlib.md5())


def fs_file_hash_sha1(path: Union[str, bytes, PathLike[str], PathLike[bytes] | int]) -> str:
    return fs_file_hash(path, hashlib.sha1())


def fs_file_hash_sha256(path: Union[str, bytes, PathLike[str], PathLike[bytes] | int]) -> str:
    return fs_file_hash(path, hashlib.sha256())


def fs_file_hash_sha512(path: Union[str, bytes, PathLike[str], PathLike[bytes] | int]) -> str:
    return fs_file_hash(path, hashlib.sha512())


# endregion fs


# region threading


@final
class LazyAttribute(Generic[T]):
    def __init__(self, factory: Optional[Callable[[], T]] = None, value: Optional[T] = None):
        super(LazyAttribute, self).__init__()
        self._factory = factory
        self._value = value
        self._is_initialized = self._is_initialized_ctor = factory is None
        self._lock = None if self._is_initialized_ctor else threading.Lock()

    @property
    def is_initialized(self) -> bool:
        if self._is_initialized_ctor:
            return True
        else:
            with self._lock:
                return self._is_initialized

    @property
    def value(self) -> T:
        if self._is_initialized_ctor:
            return self._value
        else:
            with self._lock:
                if not self._is_initialized:
                    self._value = self._factory()
                    self._is_initialized = True
                return self._value


# endregion threading


# region atomic

class AtomicIncrement(ABC, Generic[T]):
    @abstractmethod
    def next(self) -> T: raise NotImplemented


class AtomicInt(AtomicIncrement[int]):
    def __init__(self, starting_value=0):
        super(AtomicInt, self).__init__()
        self._value = int(starting_value)
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += int(1)
            return self._value


class AtomicIntNamed:
    def __init__(self, starting_value=0):
        super(AtomicIntNamed, self).__init__()
        self._starting_value = int(starting_value)
        self._names: dict[str, AtomicInt] = {}
        self._lock = threading.Lock()

    def next(self, name: str) -> int:
        with self._lock:
            d = self._names
            if name in d:
                return d[name].next()
            d[name] = a = AtomicInt(starting_value=self._starting_value)
            return a.next()


_ATOMIC_INT = AtomicInt()
_ATOMIC_INT_NAMED = AtomicIntNamed()


def next_int(name: str = None) -> int:
    return _ATOMIC_INT.next() if name is None else _ATOMIC_INT_NAMED.next(name)


# endregion atomic


# region class_helpers

class ClassInfo:
    CLASS_INSTANCE_NAME_FORMATTER: Callable[..., ClassInfo] = lambda s: s.class_name + '{' + str(
        s.class_instance_id
    ) + '}'

    def __init__(self, *args, **kwargs) -> None:
        # super(ClassInfo, self).__init__()

        if not hasattr(self, '_class_name'):
            self._class_name: str = parse_class_name(self)

        if not hasattr(self, '_module_name'):
            self._module_name: str = parse_module_name(self)

        if not hasattr(self, '_class_instance_id'):
            self._class_instance_id: int = next_int(name=self.class_name)

        super().__init__(*args, **kwargs)

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def module_name(self) -> str:
        return self._module_name

    @property
    def class_instance_id(self) -> int:
        return self._class_instance_id

    @property
    def class_instance_name(self) -> str:
        return self.__class__.CLASS_INSTANCE_NAME_FORMATTER(self)


# TLogger = TypeVar("TLogger", bound=Union[Logger, LoggerAdapter[
#     Union[
#         Logger,
#         LoggerAdapter[Logger],
#         LoggerAdapter[Any],
#         Any
#     ]
# ]])
TLogger = TypeVar('TLogger', bound=Union[Logger, LoggerAdapter])


class ClassLogging:
    _LOG_EXTRA_NAMES: List[str] = ['module_name', 'class_name', 'class_instance_id']

    def __init__(self, *args, **kwargs) -> None:
        # super(ClassLogging, self).__init__()
        if not hasattr(self, '_log_lazy'):
            self._log_lazy = LazyAttribute[TLogger](lambda: self._log_create())
        super().__init__(*args, **kwargs)

    def _log_create(self) -> TLogger:
        extra = {}
        for name in ClassLogging._LOG_EXTRA_NAMES:
            val = None
            try:
                val = getattr(self, name)
            except AttributeError:
                pass
            extra[name] = val

        log: TLogger = logger(None, self)
        if not extra:
            return log
        log = logging.LoggerAdapter(log, extra)
        return log

    @property
    def _log(self) -> TLogger:
        return self._log_lazy.value


# endregion class_helpers


# region filter

def filter_none(iterable: Iterable[T]) -> Iterable[T]:
    return filter(lambda x: x is not None, iterable)


# endregion filter


# region JSON

JSON_ROUNDING_DECIMAL_PLACES = 3


class JsonObjectType(Enum):
    STRING = 1
    INT = 2
    FLOAT = 3
    BOOLEAN = 4
    NONE = 5
    DICT = 6
    LIST = 7


class JsonObject:
    def __init__(self, value: Any | None):
        self._json_object_type: JsonObjectType
        self._value = value

        if value is None:
            self._json_object_type = JsonObjectType.NONE

    @property
    def value(self):
        return self._value

    @property
    def json_object_type(self) -> JsonObjectType:
        return self.json_object_type


class JsonDict:
    @staticmethod
    def _format_key(key: str | None):
        if key is None:
            return None
        chars = "\\`*_{}/[]()>< #+-.!$"
        for c in chars:
            if c in key:
                key = key.replace(c, "")
        return trim_casefold(key)

    FORMAT_KEY_FUNC = _format_key

    def __init__(self, json: Mapping[str, Any | None] | str | None):
        self._log = logging.getLogger(__name__)
        self._json_original: Mapping[str, Any | None]
        self._json_original_str: str | None
        self._json_key_formatted: Mapping[str, Any | None]
        self._json_key_formatted_str: str | None

        if json is None:
            self._json_original = {}
            self._json_original_str = None
        elif isinstance(json, Mapping):
            self._json_original = json
            self._json_original_str = json2str(json)
        elif isinstance(json, str):
            self._json_original = json2obj(json)
            self._json_original_str = json
        else:
            msg = f"JSON Type {type(json).__name__} is not supported  {json}"
            self._log.error(f"{self.__class__.__name__}.__init__(json={json}) -> {msg}")
            raise NotImplementedError(msg)
        d = {}
        for k, v in self._json_original.items():
            k = self.__class__.FORMAT_KEY_FUNC(k)
            if k is not None:
                d[k] = v
        self._json_key_formatted = d
        self._json_key_formatted_str = None if self._json_original_str is None else json2str(d)

        self.json_str = self._json_original_str

    def __str__(self):
        return self.json_str

    def __repr__(self):
        return self.__class__.__module__ + "." + self.__class__.__name__ + f"({self._json_original_str})"

    def get_value(self, key: str, value_type: type = None) -> Any | None:
        if key is None:
            return None
        if len(self._json_original) == 0:
            return None

        v = self._json_original.get(key)
        if v is None:
            key_formatted = self.__class__.FORMAT_KEY_FUNC(key)
            if key_formatted is not None:
                v = self._json_key_formatted.get(key)
        if v is None:
            return None
        if value_type is None:
            return v
        if value_type == str:
            return trim(xstr(v))
        elif value_type == bool:
            v = trim(xstr(v))
            return None if v is None else bool_parse(v)
        elif value_type == Decimal:
            v = trim(xstr(v))
            return None if v is None else decimal_round(Decimal(v), 3)
        elif value_type == int:
            v = trim(xstr(v))
            if v is None:
                return None
            if "." in v:
                raise ValueError(f"JSON value {key}={v} is a float not an int")
            return int(v)
        elif value_type == float:
            v = trim(xstr(v))
            return None if v is None else float(v)
        elif value_type == UUID:
            v = trim(xstr(v))
            return None if v is None else UUID(v)
        else:
            raise NotImplementedError(f"Parsing to type {value_type.__name__} is not implemented")

    def get_str(self, key: str) -> str | None:
        return self.get_value(key, str)

    def get_int(self, key: str) -> int | None:
        return self.get_value(key, int)

    def get_float(self, key: str) -> float | None:
        return self.get_value(key, float)

    def get_decimal(self, key: str) -> Decimal | None:
        return self.get_value(key, Decimal)

    def get_bool(self, key: str) -> bool | None:
        return self.get_value(key, bool)

    def get_uuid(self, key: str) -> UUID | None:
        return self.get_value(key, UUID)

    def get_list(self, key: str) -> List[Any]:
        lis = self.get_value(key)
        if lis is None:
            return []
        if not isinstance(lis, List):
            raise TypeError(f"JSON item for key '{key}' is type '{type(lis).__name__}' but not a List: {lis}")
        return lis

    def get_dict(self, key: str) -> JsonDict:
        d = self.get_value(key)
        if d is None:
            return JsonDict({})
        if not isinstance(d, Mapping):
            raise TypeError(f"JSON item for key '{key}' is type '{type(d).__name__}' but not a dict: {d}")
        return JsonDict(d)

    def items(self):
        return self._json_original.items()


# endregion JSON


# region color

# https://github.com/ubernostrum/webcolors/blob/6ac4121d44c8eda10090e8e9463ea1cf23f1932d/src/webcolors/constants.py#L17
_COLOR_HEX_REGEX = re.compile(r"^#?([a-fA-F0-9]{3}|[a-fA-F0-9]{6})$")


class Color:

    # https://github.com/ubernostrum/webcolors/blob/trunk/src/webcolors/constants.py
    # noinspection SpellCheckingInspection
    NAME_2_HEX = DictStr(
        key_converter=lambda k: k.lower(), data={
            "aliceblue": "#f0f8ff",
            "antiquewhite": "#faebd7",
            "aqua": "#00ffff",
            "aquamarine": "#7fffd4",
            "azure": "#f0ffff",
            "beige": "#f5f5dc",
            "bisque": "#ffe4c4",
            "black": "#000000",
            "blanchedalmond": "#ffebcd",
            "blue": "#0000ff",
            "blueviolet": "#8a2be2",
            "brown": "#a52a2a",
            "burlywood": "#deb887",
            "cadetblue": "#5f9ea0",
            "chartreuse": "#7fff00",
            "chocolate": "#d2691e",
            "coral": "#ff7f50",
            "cornflowerblue": "#6495ed",
            "cornsilk": "#fff8dc",
            "crimson": "#dc143c",
            "cyan": "#00ffff",
            "darkblue": "#00008b",
            "darkcyan": "#008b8b",
            "darkgoldenrod": "#b8860b",
            "darkgray": "#a9a9a9",
            "darkgrey": "#a9a9a9",
            "darkgreen": "#006400",
            "darkkhaki": "#bdb76b",
            "darkmagenta": "#8b008b",
            "darkolivegreen": "#556b2f",
            "darkorange": "#ff8c00",
            "darkorchid": "#9932cc",
            "darkred": "#8b0000",
            "darksalmon": "#e9967a",
            "darkseagreen": "#8fbc8f",
            "darkslateblue": "#483d8b",
            "darkslategray": "#2f4f4f",
            "darkslategrey": "#2f4f4f",
            "darkturquoise": "#00ced1",
            "darkviolet": "#9400d3",
            "deeppink": "#ff1493",
            "deepskyblue": "#00bfff",
            "dimgray": "#696969",
            "dimgrey": "#696969",
            "dodgerblue": "#1e90ff",
            "firebrick": "#b22222",
            "floralwhite": "#fffaf0",
            "forestgreen": "#228b22",
            "fuchsia": "#ff00ff",
            "gainsboro": "#dcdcdc",
            "ghostwhite": "#f8f8ff",
            "gold": "#ffd700",
            "goldenrod": "#daa520",
            "gray": "#808080",
            "grey": "#808080",
            "green": "#008000",
            "greenyellow": "#adff2f",
            "honeydew": "#f0fff0",
            "hotpink": "#ff69b4",
            "indianred": "#cd5c5c",
            "indigo": "#4b0082",
            "ivory": "#fffff0",
            "khaki": "#f0e68c",
            "lavender": "#e6e6fa",
            "lavenderblush": "#fff0f5",
            "lawngreen": "#7cfc00",
            "lemonchiffon": "#fffacd",
            "lightblue": "#add8e6",
            "lightcoral": "#f08080",
            "lightcyan": "#e0ffff",
            "lightgoldenrodyellow": "#fafad2",
            "lightgray": "#d3d3d3",
            "lightgrey": "#d3d3d3",
            "lightgreen": "#90ee90",
            "lightpink": "#ffb6c1",
            "lightsalmon": "#ffa07a",
            "lightseagreen": "#20b2aa",
            "lightskyblue": "#87cefa",
            "lightslategray": "#778899",
            "lightslategrey": "#778899",
            "lightsteelblue": "#b0c4de",
            "lightyellow": "#ffffe0",
            "lime": "#00ff00",
            "limegreen": "#32cd32",
            "linen": "#faf0e6",
            "magenta": "#ff00ff",
            "maroon": "#800000",
            "mediumaquamarine": "#66cdaa",
            "mediumblue": "#0000cd",
            "mediumorchid": "#ba55d3",
            "mediumpurple": "#9370db",
            "mediumseagreen": "#3cb371",
            "mediumslateblue": "#7b68ee",
            "mediumspringgreen": "#00fa9a",
            "mediumturquoise": "#48d1cc",
            "mediumvioletred": "#c71585",
            "midnightblue": "#191970",
            "mintcream": "#f5fffa",
            "mistyrose": "#ffe4e1",
            "moccasin": "#ffe4b5",
            "navajowhite": "#ffdead",
            "navy": "#000080",
            "oldlace": "#fdf5e6",
            "olive": "#808000",
            "olivedrab": "#6b8e23",
            "orange": "#ffa500",
            "orangered": "#ff4500",
            "orchid": "#da70d6",
            "palegoldenrod": "#eee8aa",
            "palegreen": "#98fb98",
            "paleturquoise": "#afeeee",
            "palevioletred": "#db7093",
            "papayawhip": "#ffefd5",
            "peachpuff": "#ffdab9",
            "peru": "#cd853f",
            "pink": "#ffc0cb",
            "plum": "#dda0dd",
            "powderblue": "#b0e0e6",
            "purple": "#800080",
            "red": "#ff0000",
            "rosybrown": "#bc8f8f",
            "royalblue": "#4169e1",
            "saddlebrown": "#8b4513",
            "salmon": "#fa8072",
            "sandybrown": "#f4a460",
            "seagreen": "#2e8b57",
            "seashell": "#fff5ee",
            "sienna": "#a0522d",
            "silver": "#c0c0c0",
            "skyblue": "#87ceeb",
            "slateblue": "#6a5acd",
            "slategray": "#708090",
            "slategrey": "#708090",
            "snow": "#fffafa",
            "springgreen": "#00ff7f",
            "steelblue": "#4682b4",
            "tan": "#d2b48c",
            "teal": "#008080",
            "thistle": "#d8bfd8",
            "tomato": "#ff6347",
            "turquoise": "#40e0d0",
            "violet": "#ee82ee",
            "wheat": "#f5deb3",
            "white": "#ffffff",
            "whitesmoke": "#f5f5f5",
            "yellow": "#ffff00",
            "yellowgreen": "#9acd32",
        }
    )

    def __init__(self, color: str | tuple[int, int, int] | tuple[float, float, float] = None) -> None:
        self._rgb: tuple[int, int, int] = (0, 0, 0)
        self._hsv: tuple[float, float, float] = (0.0, 0.0, 0.0)

        if color is None:
            return

        if isinstance(color, str):
            hex_color = self.__class__.NAME_2_HEX.get(color)
            if hex_color is not None:
                self.hex = hex_color
                return

            color_match = _COLOR_HEX_REGEX.match(color)
            if color_match is not None:
                hex_digits = color_match.group(1)
                if hex_digits is not None:
                    if len(hex_digits) == 3:
                        hex_digits = "".join(2 * s for s in hex_digits)
                    assert len(hex_digits) == 6
                    self.hex = color
                    return

            raise ValueError(f"unknown color '{color}'")

        if isinstance(color, tuple):
            if len(color) != 3:
                raise ValueError(f"invalid tuple length, expected 3 but was {len(color)}: {color}")
            x, y, z = color
            if isinstance(x, float) and isinstance(y, float) and isinstance(z, float):
                self.hsv = color
                return
            if isinstance(x, int) and isinstance(y, int) and isinstance(z, int):
                self.rgb = color
                return
            raise ValueError(f"invalid tuple types {color}, " + "  ".join([f"color[{i}]='{type(color[i]).__name__}'" for i in range(len(color))]))

        raise ValueError(f"invalid color={color} type '{type(color).__name__}'")

    # noinspection PyShadowingBuiltins
    @staticmethod
    def _min_max(value, min, max):
        if value < min:
            value = min
        if value > max:
            value = max
        return value

    def _set_hsv_from_rgb(self):
        r, g, b = self._rgb
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h = self._min_max(h * 360.0, 0.0, 360.0)
        s = self._min_max(s * 100.0, 0.0, 100.0)
        v = self._min_max(v * 100.0, 0.0, 100.0)
        self._hsv = (h, s, v)

    def _set_rgb_from_hsv(self):
        h, s, v = self._hsv
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
        r = round(self._min_max(r * 255.0, 0.0, 255.0))
        g = round(self._min_max(g * 255.0, 0.0, 255.0))
        b = round(self._min_max(b * 255.0, 0.0, 255.0))
        self._rgb = (r, g, b)

    @property
    def rgb(self) -> tuple[int, int, int]:
        return self._rgb

    @rgb.setter
    def rgb(self, rgb: tuple[int, int, int]) -> None:
        r = self._min_max(rgb[0], 0, 255)
        g = self._min_max(rgb[1], 0, 255)
        b = self._min_max(rgb[2], 0, 255)
        self._rgb = (r, g, b)
        self._set_hsv_from_rgb()

    @property
    def hsv(self) -> tuple[float, float, float]:
        return self.h, self.s, self.v

    @hsv.setter
    def hsv(self, hsv: tuple[float, float, float]) -> None:
        h = self._min_max(hsv[0], 0.0, 360.0)
        s = self._min_max(hsv[1], 0.0, 100.0)
        v = self._min_max(hsv[2], 0.0, 100.0)
        self._hsv = (h, s, v)
        self._set_rgb_from_hsv()

    @property
    def r(self):
        return self._rgb[0]

    @r.setter
    def r(self, value: int):
        self.rgb = (value, self._rgb[1], self._rgb[2])

    @property
    def g(self):
        return self._rgb[1]

    @g.setter
    def g(self, value: int):
        self.rgb = (self._rgb[0], value, self._rgb[2])

    @property
    def b(self):
        return self._rgb[2]

    @b.setter
    def b(self, value: int):
        self.rgb = (self._rgb[0], self._rgb[1], value)

    @property
    def h(self):
        return self._hsv[0]

    @h.setter
    def h(self, value: float):
        self.hsv = (value, self._hsv[1], self._hsv[2])

    @property
    def s(self):
        return self._hsv[1]

    @s.setter
    def s(self, value: float):
        self.hsv = (self._hsv[0], value, self._hsv[2])

    @property
    def v(self):
        return self._hsv[2]

    @v.setter
    def v(self, value: float):
        self.hsv = (self._hsv[0], self._hsv[1], value)

    @property
    def hex(self):
        return '#%02x%02x%02x' % (self.r, self.g, self.b)

    @hex.setter
    def hex(self, value):
        h = value.lstrip('#')
        self.rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def __str__(self):
        return self.hex

    def __repr__(self):
        return f"{self.__class__.__name__}({self.hex})"

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, Color):
            return self.__eq__(Color(other))
        return self.rgb == other.rgb


class TerminalGraphicsEffect(IntFlag):
    NONE = auto()
    RESET = auto()
    BOLD = auto()
    DIM = auto()
    ITALIC = auto()
    UNDERLINE = auto()
    BLINK = auto()
    REVERSE = auto()
    HIDDEN = auto()
    STRIKE = auto()

    @classmethod
    def items(cls) -> [Tuple[Self, str, int]]:
        return [(cls(x), cls(x).name, cls(x).value) for x in cls]

    @classmethod
    def convert(cls, value: Self | int | str | None) -> Self:
        if value is None:
            return cls.NONE
        if type(value) is type(cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            value = value.upper()
            for t, n, i in cls.items():
                if n.upper() == value:
                    return t
        return cls(value)


class TerminalGraphicsColor(StrEnum):
    BLACK = auto()
    BLACK_BRIGHT = auto()
    RED = auto()
    RED_BRIGHT = auto()
    GREEN = auto()
    GREEN_BRIGHT = auto()
    YELLOW = auto()
    YELLOW_BRIGHT = auto()
    BLUE = auto()
    BLUE_BRIGHT = auto()
    MAGENTA = auto()
    MAGENTA_BRIGHT = auto()
    CYAN = auto()
    CYAN_BRIGHT = auto()
    WHITE = auto()
    WHITE_BRIGHT = auto()
    DEFAULT = auto()

    @classmethod
    def items(cls) -> [Tuple[Self, str]]:
        return [(cls(x), cls(x).name) for x in cls]

    @classmethod
    def convert(cls, value: Self | str | None) -> Self:
        if value is None:
            return cls.DEFAULT
        if type(value).__name__ == cls.__name__:
            return value
        if isinstance(value, str):
            value = value.upper()
            for t, n in cls.items():
                if n.upper() == value:
                    return t
        return cls(value)


class TerminalGraphicsFormatter(ClassInfo):
    debug = False

    _EFFECT_2_ANSI: dict[TerminalGraphicsEffect, int] = {
        TerminalGraphicsEffect.NONE: -1000,
        TerminalGraphicsEffect.RESET: 0,
        TerminalGraphicsEffect.BOLD: 1,
        TerminalGraphicsEffect.DIM: 2,
        TerminalGraphicsEffect.ITALIC: 3,
        TerminalGraphicsEffect.UNDERLINE: 4,
        TerminalGraphicsEffect.BLINK: 5,
        TerminalGraphicsEffect.REVERSE: 7,
        TerminalGraphicsEffect.HIDDEN: 8,
        TerminalGraphicsEffect.STRIKE: 9,
    }

    _COLOR_2_ANSI: dict[TerminalGraphicsColor, int] = {
        TerminalGraphicsColor.BLACK: 30,
        TerminalGraphicsColor.BLACK_BRIGHT: 90,
        TerminalGraphicsColor.RED: 31,
        TerminalGraphicsColor.RED_BRIGHT: 91,
        TerminalGraphicsColor.GREEN: 32,
        TerminalGraphicsColor.GREEN_BRIGHT: 92,
        TerminalGraphicsColor.YELLOW: 33,
        TerminalGraphicsColor.YELLOW_BRIGHT: 93,
        TerminalGraphicsColor.BLUE: 34,
        TerminalGraphicsColor.BLUE_BRIGHT: 94,
        TerminalGraphicsColor.MAGENTA: 35,
        TerminalGraphicsColor.MAGENTA_BRIGHT: 95,
        TerminalGraphicsColor.CYAN: 36,
        TerminalGraphicsColor.CYAN_BRIGHT: 96,
        TerminalGraphicsColor.WHITE: 37,
        TerminalGraphicsColor.WHITE_BRIGHT: 97,
        TerminalGraphicsColor.DEFAULT: 39,
    }

    _PREFIX = "\033["
    _SUFFIX = "m"
    _RESET = _PREFIX + '0' + _SUFFIX

    def __init__(
        self,
        fg: TerminalGraphicsColor | str | None = None,
        bg: TerminalGraphicsColor | str | None = None,
        ef: TerminalGraphicsEffect | int | None = None,
    ):
        super(TerminalGraphicsFormatter, self).__init__()

        tgc = TerminalGraphicsColor
        tge = TerminalGraphicsEffect
        r = self._RESET

        self._fg = fg = tgc.convert(fg)
        self._bg = bg = tgc.convert(bg)
        self._ef = ef = tge.convert(ef)

        self._ansi = []
        self._prefix = ''
        self._suffix = ''

        if fg is tgc.DEFAULT and bg is tgc.DEFAULT and ef is tge.NONE:
            return

        if fg is tgc.DEFAULT and bg is tgc.DEFAULT and ef is tge.RESET:
            self._ansi.append(0)
            self._prefix += r
            return

        ps: [int] = [
            self._COLOR_2_ANSI[fg],
            self._COLOR_2_ANSI[bg] + 10,
        ]
        for e in [effect[0] for effect in tge.items()]:
            if e is not tge.NONE and e in ef:
                ps.append(self._EFFECT_2_ANSI[e])

        ps = [x for x in ps if x >= 0]
        ps.sort()

        self._ansi = ps

        self._prefix = r + self._PREFIX + ';'.join([str(s) for s in ps]) + self._SUFFIX
        self._suffix = r

    def format(self, text: str) -> str:
        t = self._prefix + text + self._suffix
        if self.debug:
            t = (self._prefix.replace("\033", "\\033")
                 + t.replace("\n", "")
                 + self._suffix.replace("\033", "\\033")
                 + "\n")
        return t

    @classmethod
    def create(cls, *args) -> TerminalGraphicsFormatter:
        if args is None or len(args) == 0:
            return TerminalGraphicsFormatter()
        formatters = []
        colors = []
        effects = []
        items = [arg for arg in args]
        while len(items) > 0:
            item = items.pop(0)
            if item is None:
                colors.append(None)
            elif isinstance(item, tuple) or isinstance(item, list):
                for x in item:
                    items.append(x)
            elif isinstance(item, TerminalGraphicsFormatter):
                formatters.append(item)
            elif isinstance(item, TerminalGraphicsColor) or isinstance(item, str):
                colors.append(TerminalGraphicsColor.convert(item))
            elif isinstance(item, TerminalGraphicsEffect) or isinstance(item, int):
                effects.append(TerminalGraphicsEffect(item))

        effects_flags = TerminalGraphicsEffect.NONE
        for effect in effects:
            effects_flags |= effect

        if len(formatters) > 0:
            return formatters[0]

        color_fg = colors[0] if 0 < len(colors) else None
        color_bg = colors[1] if 1 < len(colors) else None
        if color_fg is None:
            color_fg = TerminalGraphicsColor.DEFAULT
        if color_bg is None:
            color_bg = TerminalGraphicsColor.DEFAULT

        return TerminalGraphicsFormatter(color_fg, color_bg, effects_flags)


# endregion color


# region images

# @formatter:off

# https://stackoverflow.com/questions/73758704/is-there-a-way-to-clear-a-tree-element-in-pysimplegui-before-adding-new-data
# folder_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABnUlEQVQ4y8WSv2rUQRSFv7vZgJFFsQg2EkWb4AvEJ8hqKVilSmFn3iNvIAp21oIW9haihBRKiqwElMVsIJjNrprsOr/5dyzml3UhEQIWHhjmcpn7zblw4B9lJ8Xag9mlmQb3AJzX3tOX8Tngzg349q7t5xcfzpKGhOFHnjx+9qLTzW8wsmFTL2Gzk7Y2O/k9kCbtwUZbV+Zvo8Md3PALrjoiqsKSR9ljpAJpwOsNtlfXfRvoNU8Arr/NsVo0ry5z4dZN5hoGqEzYDChBOoKwS/vSq0XW3y5NAI/uN1cvLqzQur4MCpBGEEd1PQDfQ74HYR+LfeQOAOYAmgAmbly+dgfid5CHPIKqC74L8RDyGPIYy7+QQjFWa7ICsQ8SpB/IfcJSDVMAJUwJkYDMNOEPIBxA/gnuMyYPijXAI3lMse7FGnIKsIuqrxgRSeXOoYZUCI8pIKW/OHA7kD2YYcpAKgM5ABXk4qSsdJaDOMCsgTIYAlL5TQFTyUIZDmev0N/bnwqnylEBQS45UKnHx/lUlFvA3fo+jwR8ALb47/oNma38cuqiJ9AAAAAASUVORK5CYII='
# file_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABU0lEQVQ4y52TzStEURiHn/ecc6XG54JSdlMkNhYWsiILS0lsJaUsLW2Mv8CfIDtr2VtbY4GUEvmIZnKbZsY977Uwt2HcyW1+dTZvt6fn9557BGB+aaNQKBR2ifkbgWR+cX13ubO1svz++niVTA1ArDHDg91UahHFsMxbKWycYsjze4muTsP64vT43v7hSf/A0FgdjQPQWAmco68nB+T+SFSqNUQgcIbN1bn8Z3RwvL22MAvcu8TACFgrpMVZ4aUYcn77BMDkxGgemAGOHIBXxRjBWZMKoCPA2h6qEUSRR2MF6GxUUMUaIUgBCNTnAcm3H2G5YQfgvccYIXAtDH7FoKq/AaqKlbrBj2trFVXfBPAea4SOIIsBeN9kkCwxsNkAqRWy7+B7Z00G3xVc2wZeMSI4S7sVYkSk5Z/4PyBWROqvox3A28PN2cjUwinQC9QyckKALxj4kv2auK0xAAAAAElFTkSuQmCC'

# @formatter:on

# endregion images


# region logging


class LogColorStreamWrapper:

    def __init__(
        self,
        stream,
        debug: Any = None,
        info: Any = None,
        warning: Any = None,
        error: Any = None,
        critical: Any = None,
    ):
        self.stream = stream
        self.level_indexes: [int] = []

        tgf = TerminalGraphicsFormatter
        # tgc = TerminalGraphicsColor

        debug = tgf.create(debug)
        info = tgf.create(info)
        warning = tgf.create(warning)
        error = tgf.create(error)
        critical = tgf.create(critical)

        self.colors = {
            'DEBUG': debug,
            'INFO': info,
            'WARNING': warning,
            'ERROR': error,
            'CRITICAL': critical,
        }

    def write(self, text: Any):
        if text is None:
            return
        if not isinstance(text, str):
            text = str(text)
        formatter = self.get_formatter(text)
        if formatter is None:
            self.stream.write(text)
        else:
            self.stream.write(formatter.format(text))

    def get_formatter(self, text: str) -> TerminalGraphicsFormatter | None:
        # check cached indexes
        for ix in self.level_indexes:
            for k, v in self.colors.items():
                if text[ix:ix + len(k)] == k:
                    # print(f"-- found using cache at ix: {ix} --")
                    return v

        # search for index and keep the lowest find index
        index_lowest = -1
        for k, v in self.colors.items():
            index_found = text.find(k)
            if index_found >= 0:
                if index_lowest < 0 or index_found < index_lowest:
                    index_lowest = index_found

        # we didn't find an index
        if index_lowest < 0:
            return None

        # found index, cache and retry func now that index is cached
        self.level_indexes.append(index_lowest)
        return self.get_formatter(text)


# noinspection PyShadowingBuiltins
def logging_setup(
    level: int | str = logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(module)s.%(funcName)s %(filename)s:%(lineno)d [%(name)s]: %(message)s",
    stream=sys.stdout,
    debug: Any | None = TerminalGraphicsColor.BLUE,
    info: Any | None = TerminalGraphicsColor.DEFAULT,
    warning: Any | None = (TerminalGraphicsColor.YELLOW_BRIGHT, TerminalGraphicsEffect.BOLD),
    error: Any | None = TerminalGraphicsColor.RED_BRIGHT,
    critical: Any | None = (TerminalGraphicsColor.WHITE_BRIGHT, TerminalGraphicsColor.RED_BRIGHT, TerminalGraphicsEffect.BOLD),
):
    def level_parse(lvl):
        lvl_map = {'D': logging.DEBUG, 'I': logging.INFO, 'W': logging.WARNING, 'E': logging.ERROR, 'C': logging.CRITICAL}
        if lvl is None:
            return logging.DEBUG

        if isinstance(lvl, int):
            if lvl in lvl_map.values():
                return lvl
            raise f"Invalid log level: {lvl}"

        if isinstance(lvl, str):
            lvl_char = trim(lvl.strip()[:1].upper())
            if lvl_char is None:
                return logging.DEBUG
            lvl_val = lvl_map.get(lvl_char)
            if lvl_val is not None:
                return lvl_val
            raise f"Invalid log level: {lvl}"

        return level_parse(str(lvl))

    level_int = level_parse(level)

    logging.basicConfig(
        format=format,
        stream=LogColorStreamWrapper(
            stream=stream,
            debug=debug,
            info=info,
            warning=warning,
            error=error,
            critical=critical,
        ),
        level=level_int,
    )


# endregion logging


class FileSystemSnapshot(ClassInfo, ClassLogging):

    # region init

    def __init__(self, follow_symlinks: bool = False):
        super().__init__()
        self._is_case_sensitive: bool = RUNTIME_INFO.os.is_fs_case_sensitive
        self._all: MutableMapping[str, FileSystemEntry] = self._dict()
        self._children: MutableMapping[str, list[FileSystemEntry]] = self._dict()
        self._parents = self._dict()
        self.follow_symlinks: bool = follow_symlinks

    # endregion init

    # region method

    def _dict(self) -> MutableMapping[str, Any]:
        return dict() if self.is_case_sensitive else DictStr(key_converter=lambda x: x.casefold())

    def cache(self, path: str) -> None:
        def get_entries(p: str, cdict) -> None:
            if p in cdict:
                return  # recursion check
            des = cdict.get(p)
            if des is None:
                cdict[p] = des = []
            with os.scandir(p) as it:
                for de in it:
                    if de is not None:
                        des.append(de)
                        if de.is_dir():
                            get_entries(de.path, cdict)

        path = str(self.resolve(Path(path)))
        children: dict[str, list[DirEntry]] = {}
        get_entries(path, children)
        for parent_de, children_de in children.items():
            parent_fse = self.get(parent_de, is_path_resolved=True)
            children_fse = [self.get(de, is_path_resolved=True) for de in children_de]
            self._children[parent_fse.path_str] = children_fse

    def clear(self):
        self._all = self._dict()
        self._children = self._dict()
        self._parents = self._dict()

    def resolve(self, path: Path) -> Path:
        try:
            pp = path.expanduser()
            if pp is not None:
                path = pp
        except Exception as e:
            self._log.debug(f"error expanduser()", exc_info=e)

        try:
            pp = path.resolve()
            if pp is not None:
                path = pp
        except Exception as e:
            self._log.debug(f"error resolve()", exc_info=e)

        return path

    def add(self, entry: FileSystemEntry) -> None:
        self._all[entry.path_str] = entry

    def get(self, path: str | DirEntry, is_path_resolved: bool = False) -> FileSystemEntry | None:
        # o = self._all.get(path)
        path_str = path.path if isinstance(path, DirEntry) else path
        o = self._all.get(path_str)
        if o is not None:
            return o
        o = FileSystemEntry(path, snapshot=self, is_path_resolved=is_path_resolved)
        self._all[path_str] = o
        return o

    # noinspection PyBroadException
    def get_children(self, entry: FileSystemEntry) -> list[FileSystemEntry]:
        if not entry.is_dir:
            return []

        path_str = entry.path_str
        children = self._children.get(path_str)
        if children is not None:
            return children

        children_de = []
        try:
            with os.scandir(path_str) as it:
                for obj in it:
                    if obj is not None:
                        children_de.append(obj)

        except Exception as e:
            msg = f"Could not get directory listing for {path_str}"
            _log.warning(msg, exc_info=e)
            entry.add_error(msg, e)
            return []  # do not cache failed attempts

        children: list[FileSystemEntry] = []
        for child_de in children_de:
            children.append(self.get(child_de, is_path_resolved=True))

        self._children[path_str] = children
        return children

    # noinspection PyBroadException
    def get_parent(self, entry: FileSystemEntry) -> FileSystemEntry | None:
        path_parent = entry.path.parent
        if path_parent is None:
            return None
        path_parent = path_parent.absolute()
        if path_parent is None:
            return None

        if self._is_case_sensitive:
            if str(path_parent) == str(entry.path):
                return None  # at root so return no parent
        else:
            if str(path_parent).casefold() == str(entry.path).casefold():
                return None  # at root so return no parent

        return self.get(str(path_parent))

    def get_children_all(self, entry: FileSystemEntry, include_self: bool = False) -> list[FileSystemEntry]:
        if not entry.path.is_dir():
            return [entry] if include_self else []

        entries = self._dict()
        dirs = [entry]
        while len(dirs) > 0:
            dir_current = dirs.pop()
            for child in dir_current.children:
                child_path_str = child.path_str
                if child_path_str in entries:
                    continue  # already have the entry, so prevent infinite recursion
                entries[child_path_str] = child
                if child.is_dir:
                    dirs.append(child)

        items = [x for x in entries.values()]
        if include_self:
            items.append(entry)
        # items.sort()
        return items

    # endregion method

    # region @property

    @property
    def is_case_sensitive(self) -> bool:
        return self._is_case_sensitive

    # endregion @property

    # region override

    def __str__(self):
        return f"{self.__class__.__name__}[{len(self._all)}]"

    # endregion override


class FileSystemEntry:

    __slots__ = (
        '_snapshot',
        '_path',
        '_path_str',
        '_path_str_cased',
        '_path_parts',
        '_path_parts_cased',
        '_name',
        '_dir_entry',
        '_stat',
        '_lstat',
        '_is_file',
        '_is_dir',
        '_is_symlink',
        '_follow_symlinks',
        '_datetime_accessed',
        '_datetime_modified',
        '_datetime_created',
        '_size',
        'errors',
    )

    # region init

    def __init__(
        self,
        path: str | Path | DirEntry,
        follow_symlinks: bool = None,
        snapshot: FileSystemSnapshot = None,
        is_path_resolved: bool = False,
    ) -> None:
        super().__init__()
        self._snapshot = snapshot if snapshot is not None else FileSystemSnapshot()
        self._path: Path | None = None
        self._path_str: str | None = None
        self._path_str_cased: str | None = None
        self._path_parts: list[str] | None = None
        self._path_parts_cased: list[str] | None = None
        self._name: str | None = None
        self._dir_entry: DirEntry | None = None
        self._stat: os.stat_result | None = None
        self._lstat: os.stat_result | None = None
        self._is_file: bool | None = None
        self._is_dir: bool | None = None
        self._is_symlink: bool | None = None
        self._follow_symlinks: bool | None = follow_symlinks
        self._datetime_accessed: datetime | None = None
        self._datetime_modified: datetime | None = None
        self._datetime_created: datetime | None = None
        self._size: int | None = None
        # self._children: list[DirEntry] | None = None
        self.errors: list[(str, Exception | None)] = []

        if isinstance(path, DirEntry):
            self._dir_entry = path
            if is_path_resolved:
                self._path_str = path.path
            else:
                path = Path(path.path)
                _log.debug(f"resolving path: {path}")
                path = self._snapshot.resolve(path)
                self._path = path
                self._path_str = str(path)

        elif isinstance(path, str):
            if is_path_resolved:
                self._path_str = path
            else:
                path = Path(path)
                _log.debug(f"resolving path: {path}")
                path = self._snapshot.resolve(path)
                self._path = path
                self._path_str = str(path)

        elif isinstance(path, Path):
            if is_path_resolved:
                self._path = path
                self._path_str = str(path)
            else:
                _log.debug(f"resolving path: {path}")
                path = self._snapshot.resolve(path)
                self._path = path
                self._path_str = str(path)

        self._path_str_cased = self._path_str if self._snapshot.is_case_sensitive else self._path_str.casefold()

    # endregion init

    # region method

    def add_error(self, msg: str, exception: Exception | None = None) -> None:
        self.errors.append((msg, exception))

    # endregion method

    # region @property

    @property
    def path_parts(self) -> list[str]:
        o = self._path_parts
        if o is None:
            o = [x for x in self.path.parts]
            self._path_parts = o
        return o

    @property
    def path_parts_cased(self) -> list[str]:
        o = self._path_parts_cased
        if o is None:
            o = [(x if self._snapshot.is_case_sensitive else x.casefold()) for x in self.path_parts]
            self._path_parts_cased = o
        return o

    @property
    def follow_symlinks(self) -> bool:
        o = self._follow_symlinks
        if o is None:
            o = self._snapshot.follow_symlinks
        return o

    @follow_symlinks.setter
    def follow_symlinks(self, value: bool | None):
        self._follow_symlinks = value

    @property
    def path_str(self) -> str:
        return self._path_str

    @property
    def path(self) -> Path:
        o = self._path
        if o is None:
            o = Path(self._path_str)
            self._path = o
        return o

    @property
    def name(self) -> str:
        o = self._name
        if o is None:
            o = self.path.name
            self._name = o
        return o

    @property
    def dir_entry(self) -> DirEntry | None:
        return self._dir_entry

    @property
    def stat(self) -> os.stat_result:
        o = self._stat
        if o is None:
            if self._dir_entry is not None:
                o = self._dir_entry.stat(follow_symlinks=self.follow_symlinks)
            else:
                o = os.stat(self.path, follow_symlinks=self.follow_symlinks)
            self._stat = o
        return o

    @property
    def lstat(self) -> os.stat_result:
        o = self._lstat
        if o is None:
            o = os.lstat(self.path)
            self._lstat = o
        return o

    @property
    def is_file(self) -> bool:
        o = self._is_file
        if o is None:
            if self._dir_entry is not None:
                o = self._dir_entry.is_file(follow_symlinks=self.follow_symlinks)
            else:
                o = self.path.is_file()
            self._is_file = o
        return o

    @property
    def is_dir(self) -> bool:
        o = self._is_dir
        if o is None:
            if self._dir_entry is not None:
                o = self._dir_entry.is_dir(follow_symlinks=self.follow_symlinks)
            else:
                o = self.path.is_dir()
            self._is_dir = o
        return o

    @property
    def is_symlink(self) -> bool:
        o = self._is_symlink
        if o is None:
            if self._dir_entry is not None:
                o = self._dir_entry.is_symlink()
            else:
                o = self.path.is_symlink()
            self._is_symlink = o
        return o

    # noinspection PyBroadException
    @property
    def datetime_accessed(self) -> datetime:
        o = self._datetime_accessed
        if o is None:
            if o is None:
                try:
                    o = datetime.fromtimestamp(self.stat.st_atime, tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                try:
                    o = datetime.fromtimestamp(os.path.getatime(self.path), tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                msg = f"Could not get datetime_accessed for {self.path_str}"
                logger().warning(msg)
                self.add_error(msg)
                o = datetime.min.astimezone(tz=timezone.utc)

            self._datetime_accessed = o
        return o

    # noinspection PyBroadException
    @property
    def datetime_modified(self) -> datetime:
        o = self._datetime_modified
        if o is None:
            if o is None:
                try:
                    o = datetime.fromtimestamp(self.stat.st_mtime, tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                try:
                    o = datetime.fromtimestamp(os.path.getmtime(self.path), tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                msg = f"Could not get datetime_modified for {self.path_str}"
                logger().warning(msg)
                self.add_error(msg)
                o = datetime.min.astimezone(tz=timezone.utc)

            self._datetime_modified = o
        return o

    # noinspection PyBroadException
    @property
    def datetime_created(self) -> datetime:
        o = self._datetime_created
        if o is None:
            try:
                if RUNTIME_INFO.os in [OperatingSystem.WINDOWS, OperatingSystem.WINDOWS_CYGWIN]:
                    o = datetime.fromtimestamp(self.stat.st_ctime, tz=timezone.utc)
                else:
                    try:
                        o = datetime.fromtimestamp(self.stat.st_birthtime, tz=timezone.utc)
                    except AttributeError:
                        o = datetime.fromtimestamp(self.stat.st_mtime, tz=timezone.utc)
            except Exception:
                pass

            if o is None:
                try:
                    o = datetime.fromtimestamp(self.stat.st_ctime, tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                try:
                    o = datetime.fromtimestamp(self.stat.st_birthtime, tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                try:
                    o = datetime.fromtimestamp(self.stat.st_mtime, tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                try:
                    o = datetime.fromtimestamp(os.path.getctime(self.path), tz=timezone.utc)
                except Exception:
                    pass

            if o is None:
                msg = f"Could not get datetime_created for {self.path_str}"
                logger().warning(msg)
                self.add_error(msg)
                o = datetime.min.astimezone(tz=timezone.utc)

            self._datetime_created = o
        return o

    # noinspection PyBroadException
    @property
    def size(self) -> int:
        o = self._size
        if o is None:
            try:
                if self.is_file:
                    o = self.stat.st_size
                elif self.is_symlink:
                    o = self.lstat.st_size
                elif self.is_dir:
                    o = self.stat.st_size
            except Exception:
                pass

            if o is None or o < 0:
                try:
                    o = os.path.getsize(self.path)
                except Exception:
                    pass

            if o is None or o < 0:
                try:
                    o = self.stat.st_size
                except Exception:
                    pass

            if o is None or o < 0:
                try:
                    o = self.lstat.st_size
                except Exception:
                    pass

            if o is None:
                msg = f"Could not get size for {self.path_str}"
                logger().warning(msg)
                self.add_error(msg)
                o = -1

            self._size = o
        return o

    @property
    def children(self) -> [FileSystemEntry]:
        return self._snapshot.get_children(self)

    @property
    def parent(self) -> FileSystemEntry | None:
        return self._snapshot.get_parent(self)

    @property
    def children_all(self) -> [FileSystemEntry]:
        return self._snapshot.get_children_all(self)

    # endregion @property

    # region override

    def __eq__(self, __value):
        if __value is None:
            return False
        if not isinstance(__value, FileSystemEntry):
            return self.__eq__(self.__class__(str(__value)))
        return self._path_str_cased == __value._path_str_cased

    def __str__(self):
        return self._path_str

    def __hash__(self):
        return hash(self._path_str_cased)

    def __lt__(self, __value):
        if __value is None:
            return False

        if not isinstance(__value, FileSystemEntry):
            return self.__lt__(self.__class__(str(__value)))

        return compare_iterable(self.path_parts_cased, __value.path_parts_cased)

    # endregion override
