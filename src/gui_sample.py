from src.mrs.mrs_gui import *

# noinspection PyPep8Naming

_log = logger(__name__)


def run():

    sg.theme('LightGrey2')
    sg.set_options(
        suppress_error_popups=True,
        suppress_raise_key_errors=False,
        suppress_key_guessing=True,
        warn_button_key_duplicates=True,
    )

    window_key = WindowKey('window1')

    r = TreeRow
    data = [
        r('', 'k1', None, 't1', 'v11', 'v12', 'v13', 'v14'),
        r('', 'k2', None, 't2', 'v21', 'v22', 'v23', 'v24'),
        r('k2', 'k2a', None, 't2a', 'v21a', 'v22a', 'v23a', 'v24a'),
        r('k2a', 'k2a1', None, 't2a1', 'v21a1', 'v22a1', 'v23a1', 'v24a1'),
        r('k2a', 'k2a2', None, 't2a2', 'v21a2', 'v22a2', 'v23a2', 'v24a2'),
        r('k2a', 'k2a3', None, 't2a3', 'v21a3', 'v22a3', 'v23a3', 'v24a3'),
        r('k2a3', 'k2a3a', None, 't2a3aRRRRRRRRRRRR', 'v21a3a', 'v22a3a', 'v23a3a', 'v24a3a'),
        r('k2', 'k2b', None, 't2b', 'v21b', 'v22b', 'v23b', 'v24b'),
    ]

    c_files = ColumnCollapsable(
        key=window_key.child('files'),
        background_color=WindowColor.GREEN_LIGHT,
        layout=[[Tree(
            key=window_key.child('files', 'tree'),
            column_names=['Text', 'V1', 'V2', 'V3', 'V4'],
            rows=data,
            select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
            num_rows=20,
            justification='left',
            show_expanded=True,
            enable_events=True,
            def_col_width=10,
            expand_x=True,
            auto_size_columns=True,
            background_color=WindowColor.RED_LIGHT
        )]]
    )

    c_files2 = ColumnCollapsable(
        key=window_key.child('files2'),
        background_color=WindowColor.GREEN_LIGHT,
        layout=[[Tree(
            key=window_key.child('files2', 'tree'),
            column_names=['Text', 'V1', 'V2', 'V3', 'V4'],
            rows=data,
            select_mode=sg.TABLE_SELECT_MODE_EXTENDED,
            num_rows=20,
            justification='left',
            show_expanded=True,
            enable_events=True,
            def_col_width=10,
            expand_x=True,
            auto_size_columns=True,
            background_color=WindowColor.YELLOW_LIGHT
        )]]
    )

    dir_key = window_key.child('dir')

    c_browsedir = ColumnBrowseDir(
        key=dir_key.child('browse'),
        show_recursive_checkbox=True,
        default_directory='~/temp',
        default_recursive_checked=True,
    )

    def dir_cache_scan(event: WindowEvent):
        print('Scanning directory')

    def dir_cache_clear(event: WindowEvent):
        print('Clearing scan cache')

    c_browsedir_scan = sg.Column(
        key=dir_key.child('cache'),
        layout=[[
            sg.Text(key=dir_key.child('cache', 'label'), text='0 files in cache'),
            ButtonWindowEvent(key=dir_key.child('cache', 'scan'), on_window_event=dir_cache_scan),
            ButtonWindowEvent(key=dir_key.child('cache', 'clear'), on_window_event=dir_cache_clear),
        ]]
    )

    w = Window(
        title='SampleWindow',
        layout=[
            [c_files],
            [c_files2],
            [c_browsedir],
            [c_browsedir_scan],
        ],
        font=('Ariel', 12),
        resizable=True,
        size=(800, 600),
    )

    # w.Finalize()
    # w.BringToFront()
    # w.bind('<Configure>', '_config_change_')

    w.run()
    w.close()
