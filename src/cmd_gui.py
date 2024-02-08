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

from src.mrs.mrs_gui import *

_log = logger(__name__)


class FileSystemManager(Window):
    def __init__(self):
        super(FileSystemManager, self).__init__()
        self.directory: WindowElementBrowseDirEvent
        self.file_system = FileSystemEntrySnapshot()
        self.tree_data = []


def run():
    # sg.theme_previewer()
    # sg.preview_all_look_and_feel_themes()
    window = FileSystemManager()
    window.theme = 'LightGrey1'  # 'dark'

    dir_starting = '/Users/user/dev'
    dir_starting = '/Users/user/temp'
    dir_key = window.key.sub('dir')
    window.directory = WindowElementBrowseDirEvent.create(dir_key, dir_starting, is_recursive=False)

    dir_action_scan_key = dir_key.sub('action', 'scan')
    dir_action_clear_key = dir_key.sub('action', 'clear')
    dir_action_results_key = dir_key.sub('action', 'results')

    def dir_browse_callback(dir_event: WindowElementBrowseDirEvent):
        window.directory = dir_event
        _log.debug(f"window[{dir_action_scan_key}].update(disabled={not dir_event.is_valid})")
        window.get_element(dir_action_scan_key).update(disabled=not dir_event.is_valid)

    def dir_action_results_update():
        window.get_element(dir_action_results_key).update(f"{len(window.tree_data)} files scanned")

        td = sg.TreeData()
        for row in window.tree_data:
            td.Insert(row[0], row[1], row[2], row[3:])
        window.get_element(dir_key.sub("FileList", "Tree")).update(td)

    def dir_action_scan_callback(event: WindowEvent):
        if window.directory is None or window.directory.directory_path is None or not window.directory.is_valid:
            return
        path = str(window.directory.directory_path.resolve())

        data = []
        root = window.file_system.get(path)
        data.append(['', root.path_str, root.name, root.size])
        for entry in root.children_all:
            data.append(
                [
                    entry.parent.path_str,
                    entry.path_str,
                    entry.name,
                    entry.name,
                    entry.size,
                    'F' if entry.is_file else 'D' if entry.is_dir else 'L'
                ]
            )
        window.tree_data = data

        _log.debug(f"Scanned items: {len(window.tree_data)}")
        dir_action_results_update()

    def dir_action_clear_callback(event: WindowEvent):
        window.fs_items = []
        dir_action_results_update()

    dir_layout = [
        window_element_browse_dir_create(
            window=window,
            key_browse_dir=dir_key.sub('browse'),
            change_callback=dir_browse_callback,
            default_directory=window.directory.directory,
            default_recursive_checked=window.directory.is_recursive,
        ) + [
            sg.Button(
                button_text='Scan',
                disabled=not window.directory.is_valid,
                enable_events=True,
                key=dir_action_scan_key,
                disabled_button_color=('#666666', '#cccccc'),
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

    # font = ('Courier New', 11)
    headings = ['Name2', 'Size', 'Type']
    # col_widths = list(map(lambda x: len(x) + 2, headings))
    # max_col_width = len('ParameterNameToLongToFitIntoAColumn') + 2
    # char_width = sg.Text.char_width_in_pixels(font)
    collapsible_file_list = window_element_column_collapsible_create(
        window=window,
        key_column=dir_key.sub("FileList"),
        layout_items=[[
            sg.Tree(
                data=sg.TreeData(),
                headings=headings,
                visible_column_map=[True, True, True],
                select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
                num_rows=20,
                # col0_width=0,
                col0_heading='Key',

                justification='left',
                max_col_width=9999,
                key=dir_key.sub("FileList", "tree"),
                show_expanded=True,
                enable_events=True,
                def_col_width=20,
                # expand_x=True,
                # font=font,
                auto_size_columns=False,
                # expand_y=True,
            )
        ]],
        title_text="File List",
    )

    collapsible_remove_prefix = window_element_column_collapsible_create(
        window=window,
        key_column=dir_key.sub("RemovePrefixKey"),
        layout_items=[[sg.Button("Some Button")]],
        title_text="Remove Prefix",
    )

    collapsible_remove_suffix = window_element_column_collapsible_create(
        window=window,
        key_column=dir_key.sub("RemoveSuffixKey"),
        layout_items=[[sg.Button("Some Button 2")]],
        title_text="Remove Suffix",
    )

    window.layout = [
        # [window_element_theme_sample_list(window=window, key=dir_key.sub('themes'))], [sg.HSep(pad=(0, 10))],
        [collapsible_file_list],
        [collapsible_remove_prefix],
        [collapsible_remove_suffix],
        [sg.HSep(pad=(0, 4))],
        dir_layout
    ]

    window.subscribe(dir_action_scan_key, dir_action_scan_callback)
    window.subscribe(dir_action_clear_key, dir_action_clear_callback)

    def resize_stuff(event: WindowEvent):
        expand = 'x'
        expand_x = True
        expand_y = False
        for k, ele in window._psgwindow.key_dict.items():
            _log.debug(f"Resizing: {k}")
            try:
                if hasattr(ele, 'Widget') and ele.Widget is not None and hasattr(ele.Widget, 'expand'):
                    ele.Widget.expand(expand_x, expand_y)
            except Exception as e:
                _log.error(f"Error while ele.Widget.expand: {k}", exc_info=e)

            try:
                if hasattr(ele, 'Widget') and ele.Widget is not None and hasattr(ele.Widget, 'pack'):
                    ele.Widget.pack(expand=True, fill=expand)
            except Exception as e:
                _log.error(f"Error while ele.Widget.pack: {k}", exc_info=e)

            try:
                if hasattr(ele, 'ParentRowFrame') and ele.ParentRowFrame is not None and hasattr(ele.ParentRowFrame, 'pack'):
                    ele.ParentRowFrame.pack(expand=True, fill=expand)
            except Exception as e:
                _log.error(f"Error while ele.ParentRowFrame.pack: {k}", exc_info=e)

            try:
                if hasattr(ele, 'element_frame') and ele.element_frame is not None and hasattr(ele.element_frame, 'pack'):
                    ele.element_frame.pack(expand=True, fill=expand)
            except Exception as e:
                _log.error(f"Error while ele.element_frame.pack: {k}", exc_info=e)

            try:
                if hasattr(ele, 'TKTreeview') and ele.TKTreeview is not None and hasattr(ele.TKTreeview, 'pack'):
                    ele.TKTreeview.pack(expand=True, fill=expand)
            except Exception as e:
                _log.error(f"Error while ele.TKTreeview.pack: {k}", exc_info=e)

    def window_handler_before(event: WindowEvent):
        _log.debug(f"window_handler_before: {event}")

    def window_handler_after(event: WindowEvent):
        _log.debug(f"window_handler_after: {event}")

    def window_handler_resize2(event: WindowEvent):
        _log.debug(f"window_handler_resize2: {event}")

        resize_table(
            table=window.get_element(dir_key.sub("FileList", "tree")),
            total_width=window.get_element(dir_key.sub("FileList", "column")).get_size()[0],
            column_sizes=[10, None, 100, 50]
        )

    def window_handler_resize(event: WindowEvent):
        _log.debug(f"window_handler_resize: {event}")

        table = window.get_element(dir_key.sub("FileList", "tree"))
        table_widget = table.Widget

        def display_sizes():
            print(f"size[Window]: {window.size}")
            print(f"size[column]: " + str(window.get_element(dir_key.sub("FileList", "column")).get_size()))
            print(f"size[section_column]: " + str(window.get_element(dir_key.sub("FileList", "section_column")).get_size()))
            print(f"size[tree]: " + str(window.get_element(dir_key.sub("FileList", "tree")).get_size()))

        display_sizes()
        _log.debug('table.expand(expand_x=True, expand_y=True)')
        table.expand(expand_x=True, expand_y=True)

        # for cid in headings:
        #    table_widget.column(cid, stretch=True)
        col_size_x = window.get_element(dir_key.sub("FileList", "column")).get_size()[0]
        col_size_x = col_size_x - 100 - 50 - 38

        display_sizes()
        _log.debug('table_widget.pack_forget()')
        table_widget.pack_forget()
        display_sizes()

        _log.debug(f"table_widget.column('#0', width={col_size_x})")
        table_widget.column('#0', width=col_size_x)

        _log.debug("table_widget.column('Size', width=100)")
        table_widget.column('Size', width=100)

        _log.debug("table_widget.column('Type', width=50)")
        table_widget.column('Type', width=50)

        _log.debug("table_widget.pack(side='left', fill='both', expand=True)")
        table_widget.pack(side='left', fill='both', expand=True)

        display_sizes()

    window.handler_before = window_handler_before
    window.handler_after = window_handler_after
    window.subscribe(window.key_config_size, window_handler_resize2)
    window.start()

    _log.debug("Saving settings")
    # TODO: Save directory
