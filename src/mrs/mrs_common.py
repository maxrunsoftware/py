from __future__ import annotations

import hashlib
import inspect
import logging
import os
import re
import sys
import threading
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from inspect import FrameInfo
from logging import LoggerAdapter, Logger
from os import PathLike, stat_result, DirEntry
from pathlib import Path, PurePath
from types import ModuleType, FrameType
from typing import *
from typing import Any

_log = logging.getLogger('mrs_common')

T = TypeVar('T')


# region util

def trim(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = s.strip()
    if len(s) == 0: return None
    return s


def split(delimiters: [str], string: str, maxsplit: int = 0) -> list[str]:
    """
    Split a string on multiple characters
    https://stackoverflow.com/a/13184791
    :param delimiters: the delimiters to split the string on
    :param string: the string to actually split
    :param maxsplit: the number of times to split the string. 0 means no limit
    :return: a list of strings split
    """
    regex_pattern = '|'.join(map(re.escape, delimiters))
    return re.split(regex_pattern, string, maxsplit)


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
                if frame_back is not None: return frame_back
    except Exception:
        pass

    try:
        raise Exception
    except Exception:
        try:
            frame_traceback = sys.exc_info()[2].tb_frame
            if frame_traceback is not None:
                frame_back = frame_traceback.f_back
                if frame_back is not None: return frame_back
        except Exception:
            pass

    try:
        frame_inspect = inspect.currentframe()
        if frame_inspect is not None:
            frame_back = frame_traceback.f_back
            if frame_back is not None: return frame_back
    except Exception:
        pass

    # Just return the current frame if we have it
    if frame_inspect is not None: return frame_inspect
    if frame_traceback is not None: return frame_traceback
    if frame_sys is not None: return frame_sys

    # or ANY frame
    frame_stack = inspect.stack()
    for f in frame_stack:
        if f is not None and f.frame is not None: return f.frame

    return None


_stack_current_file = os.path.normcase(trim.__code__.co_filename)


def _stack_get_first_frame_not_in_this_file() -> Optional[FrameType]:
    def is_internal_frame(_frame):
        if _frame is None: return False
        filename = os.path.normcase(_frame.f_code.co_filename)
        return filename == _stack_current_file or ('importlib' in filename and '_bootstrap' in filename)

    frame = stack_currentframe()
    while is_internal_frame(frame):
        frame_next = frame.f_back
        if frame_next == frame: return None  # prevent infinite loop
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
    if isvalid(name): return name
    module_type = sys.modules.get('__main__')
    if module_type is not None:
        name = trim(module_type.__name__)
        if isvalid(name): return name
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
                if isvalid(name): return name

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


# endregion util


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


def __fs_datetime_parse(time: float) -> datetime: return datetime.fromtimestamp(time, tz=timezone.utc)


def fs_accessed(stat: stat_result) -> datetime: return __fs_datetime_parse(stat.st_atime)


def fs_modified(stat: stat_result) -> datetime: return __fs_datetime_parse(stat.st_mtime)


def fs_created(stat: stat_result) -> datetime:
    if RUNTIME_INFO.os in [OperatingSystem.WINDOWS, OperatingSystem.WINDOWS_CYGWIN]:
        return __fs_datetime_parse(stat.st_ctime)
    else:
        try:
            return __fs_datetime_parse(stat.st_birthtime)
        except AttributeError:
            return __fs_datetime_parse(stat.st_mtime)


def fs_size(stat: stat_result) -> int: return stat.st_size


def fs_list(
        path: str,
        recursive: bool = False,
        follow_symlinks: bool = False,
        include_files: bool = True,
        include_dirs: bool = True,
        include_symlinks: bool = False,
        sort_items: bool = True
) -> list[DirEntry]:
    """Gets DirEntry objects for given directory."""
    items = []
    if not include_files and not include_dirs: return items
    _log.debug(f"fs_list({repr(path)})")

    def add_item(de: DirEntry) -> bool:
        if de is None: return False
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
            if dir_current in dirs_parsed: continue
            dirs_parsed.add(dir_current)
            try:
                with os.scandir(str(dir_current)) as it:
                    for entry in it:
                        if add_item(entry) and entry.path not in dirs_parsed:
                            dirs_stack.append(entry.path)
            except OSError:
                _log.exception(f"Error os.scandir for path: {dir_current}")

    if sort_items: items = sorted(items, key=lambda item: PurePath(item))
    return items


def fs_file_read_chunked(stream: IO[bytes], buffersize: Optional[int] = None) -> Iterator[bytes]:
    """
    Lazy function (generator) to read a file piece by piece.
    https://stackoverflow.com/a/519653
    """
    bs = buffersize if buffersize is not None else FS_BUFFERSIZE
    while True:
        data = stream.read(bs)
        if not data: break
        yield data


def fs_file_hash(
        path: Union[str, bytes, PathLike[str], PathLike[bytes] | int],
        hasher,
        result_hex: bool = True
) -> Union[bytes, str]:
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
            if name in d: return d[name].next()
            d[name] = a = AtomicInt(starting_value=self._starting_value)
            return a.next()


_ATOMIC_INT = AtomicInt()
_ATOMIC_INT_NAMED = AtomicIntNamed()


def next_int(name: str = None) -> int: return _ATOMIC_INT.next() if name is None else _ATOMIC_INT_NAMED.next(name)


# endregion atomic


# region class_helpers

class ClassInfo:
    CLASS_INSTANCE_NAME_FORMATTER: Callable[..., ClassInfo] = lambda s: s.class_name + '{' + str(
        s.class_instance_id) + '}'

    def __init__(self):
        super(ClassInfo, self).__init__()

        if not hasattr(self, '_class_name'):
            self._class_name: str = parse_class_name(self)

        if not hasattr(self, '_module_name'):
            self._module_name: str = parse_module_name(self)

        if not hasattr(self, '_class_instance_id'):
            self._class_instance_id: int = next_int(name=self.class_name)

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

    def __init__(self):
        super(ClassLogging, self).__init__()
        if not hasattr(self, '_log_lazy'): self._log_lazy = LazyAttribute[TLogger](lambda: self._log_create())

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
        if not extra: return log
        log = logging.LoggerAdapter(log, extra)
        return log

    @property
    def _log(self) -> TLogger:
        return self._log_lazy.value


# endregion class_helpers


# region filter

def filter_none(iterable: Iterable[T]) -> Iterable[T]: return filter(lambda x: x is not None, iterable)

# endregion filter


