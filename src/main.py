
from mrs.mrs_gui import *

import logging


_log = logger(__name__)


def handle_scan(self, event: WindowEvent, directory: Path, is_recursive: bool):
    self._log.debug(f"Handling scan [{is_recursive=}]: {directory}")
    entries = fs_list(str(directory), recursive=is_recursive)

    for entry in entries:
        s = "D" if entry.is_dir() else " "
        s += " "
        s += entry.path
        print(s)




def run_test():
    window = Window()
    window.title = "File System Manager"

    collapsible_remove_prefix = create_column_collapsible(
        window=window,
        layout_items=[[sg.B("Some Button")]],
        title_text="Remove Prefix",
    )
    window.layout.append([collapsible_remove_prefix])
    window.layout.append([sg.HSep(pad=(0, 20))])

    collapsible_remove_suffix = create_column_collapsible(
        window=window,
        layout_items=[[sg.B("Some Button 2")]],
        title_text="Remove Suffix",
    )
    window.layout.append([collapsible_remove_suffix])
    window.layout.append([sg.HSep(pad=(0, 20))])

    window.start()

    _log.debug("Saving settings")
    # TODO: Save directory

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(module)s.%(funcName)s %(filename)s:%(lineno)d [%(name)s]: %(message)s"
    )

    RUNTIME_INFO.log_runtime_info(_log)
    run_test()



if __name__ == '__main__':
    main()
