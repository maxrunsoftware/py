from src.mrs.mrs_common import *
from src.mrs.mrs_gui import *
import PySimpleGUI as sg

import logging

_log = logger(__name__)


def run():
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

    scan_button_key, scan_result_key = window.create_keys('scan_button_key', 'scan_result_key')


    def update_browse_dir(value: BrowseDirValues):
        window.browse_dir = value
        window[scan_button_key].enabled = value.is_valid


    browse_dir = create_elements_browse_dir(
        window=window,
        change_callback=update_browse_dir,
        default_directory='/Users/user/dev',
    )
    window.layout.append([browse_dir])

    scan_button = sg.B("Scan", key=scan_button_key)
    scan_result = sg.Text(
        text="0 Files Scanned",
        key=scan_result_key,
        size=(20, 1),
        justification="center",
    )

    def dir_scan(event: WindowEvent):
        browse_dir = getattr(window, 'browse_dir', None)
        dir = None if browse_dir is None else browse_dir.directory
        window[scan_result_key].update(f'Scanning {dir}')

    window.subscribe(scan_button_key, dir_scan)

    window.layout.append([scan_button, scan_result])

    window.start()

    _log.debug("Saving settings")
    # TODO: Save directory
