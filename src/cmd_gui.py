from src.mrs.mrs_gui import *

_log = logger(__name__)


class FileSystemManager(Window):
    def __init__(self):
        super(FileSystemManager, self).__init__()
        self.directory: WindowElementBrowseDirEvent
        self.fs_items = [0]


def run():
    window = FileSystemManager()
    dir_key = window.key.sub('dir')
    window.directory = WindowElementBrowseDirEvent.create(dir_key, '/Users/user/dev', is_recursive=False)

    dir_browse_key = dir_key.sub('browse')
    dir_action_key = dir_key.sub('action')
    dir_action_scan_key = dir_action_key.sub('scan')
    dir_action_clear_key = dir_action_key.sub('clear')
    dir_action_results_key = dir_action_key.sub('results')

    def dir_browse_callback(dir_event: WindowElementBrowseDirEvent):
        window.directory = dir_event
        _log.debug(f"window[{dir_action_scan_key}].update(disabled={not dir_event.is_valid})")
        window[dir_action_scan_key].update(disabled=not dir_event.is_valid)

    def dir_action_results_update():
        window[dir_action_results_key].update(f"{window.fs_items[0]} files scanned")

    def dir_action_scan_callback(event: WindowEvent):
        window.fs_items[0] = window.fs_items[0] + 1
        dir_action_results_update()

    def dir_action_clear_callback(event: WindowEvent):
        window.fs_items[0] = 0
        dir_action_results_update()

    dir_layout = [
        window_element_browse_dir_create(
            window=window,
            key_browse_dir=dir_browse_key,
            change_callback=dir_browse_callback,
            default_directory=window.directory.directory,
            default_recursive_checked=window.directory.is_recursive,
        ),
        [
            sg.Button(
                button_text='Scan',
                disabled=not window.directory.is_valid,
                enable_events=True,
                key=dir_action_scan_key,
            ),
            sg.Button(
                button_text='Clear',
                enable_events=True,
                key=dir_action_clear_key,
            ),
            sg.Text(
                text="0 Files Scanned",
                key=dir_action_results_key,
                size=(20, 1),
                justification="center",
            ),
        ],
    ]

    collapsible_remove_prefix_key = dir_key.sub("RemovePrefixKey")
    collapsible_remove_prefix = window_element_column_collapsible_create(
        window=window,
        key_column=collapsible_remove_prefix_key,
        layout_items=[[sg.Button("Some Button")]],
        title_text="Remove Prefix",
    )
    window.layout.append([collapsible_remove_prefix])
    window.layout.append([sg.HSep(pad=(0, 20))])

    collapsible_remove_suffix_key = dir_key.sub("RemoveSuffixKey")
    collapsible_remove_suffix = window_element_column_collapsible_create(
        window=window,
        key_column=collapsible_remove_suffix_key,
        layout_items=[[sg.Button("Some Button 2")]],
        title_text="Remove Suffix",
    )
    window.layout.append([collapsible_remove_suffix])
    window.layout.append([sg.HSep(pad=(0, 20))])

    window.layout.append(dir_layout)

    window.subscribe(dir_action_scan_key, dir_action_scan_callback)
    window.subscribe(dir_action_clear_key, dir_action_clear_callback)

    window.start()

    _log.debug("Saving settings")
    # TODO: Save directory
