from __future__ import annotations

from mrs.mrs_gui import *
from src import cmd_gui, gui_sample


def other():

    path = '/Users/user/dev'
    fses = FileSystemEntrySnapshot()
    fse = FileSystemEntry(path, snapshot=fses)
    items = [fse]
    items_children = fse.children_all
    items.extend(items_children)

    items.sort()
    for entry in items:
        if entry.is_file:
            continue

        type = ' '
        if entry.is_dir:
            type = 'D'
        elif entry.is_symlink:
            type = 'L'

        print(f"[{type}] {entry}  {len(entry.children)}")


def main():
    logging_setup()
    _log = logger(__name__)
    RUNTIME_INFO.log_runtime_info(_log)

    debug_test = True
    if debug_test:
        gui_sample.run()
    else:
        cmd_gui.run()


if __name__ == '__main__':
    main()
