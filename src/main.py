
from mrs.mrs_gui import *

import logging

from src import cmd_gui

_log = logger(__name__)


def handle_scan(self, event: WindowEvent, directory: Path, is_recursive: bool):
    self._log.debug(f"Handling scan [{is_recursive=}]: {directory}")
    entries = fs_list(str(directory), recursive=is_recursive)

    for entry in entries:
        s = "D" if entry.is_dir() else " "
        s += " "
        s += entry.path
        print(s)






def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(module)s.%(funcName)s %(filename)s:%(lineno)d [%(name)s]: %(message)s"
    )

    RUNTIME_INFO.log_runtime_info(_log)
    cmd_gui.run()



if __name__ == '__main__':
    main()
