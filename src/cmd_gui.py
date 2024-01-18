from src.cmds.command_base import CommandBase
from src.mrs.mrs_common import *
from src.mrs.mrs_gui import *
import PySimpleGUI as sg

import logging

_log = logger(__name__)

def run():
    window = Window()
    window.title = "File System Manager"
    cmds = []
    cmd_classes = CommandBase.__subclasses__()
    _log.info(f"Found {len(cmd_classes)} Commands")
    for i, cmd_class in enumerate(cmd_classes):
        _log.debug(f"cmd_classes[{i}]={cmd_class.__name__}")
        cmds.append(cmd_class(window))

    for cmd in cmds:
        cmd.add_to_window()
        window.layout.append([sg.HSep(pad=(0, 20))])

    # kdebug = "-MAIN.KDEBUG-"
    # kdebugtext = "-MAIN.KDEBUG_TEXT-"
    # window.layout.append([sg.Checkbox("Debug", key=kdebug, enable_events=True), sg.Push(), sg.Exit(size=(20, 1))])
    # window.layout.append([sg.Multiline(key=kdebugtext)])
    # window = sg.Window('Image Sorter', layout, font=font)
    # for cmd in cmds: cmd.window = window

    window.start()

    _log.debug("Saving settings")
    # TODO: Save directory


def run2():
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
