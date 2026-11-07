"""Shared Streamlit sidebar filter state across multipage navigation."""

import streamlit as st


FILTER_NAMES = ("years", "months", "accounts", "include_categories", "exclude_categories")


def _save_filter(name):
    st.session_state[f"saved_filter_{name}"] = st.session_state.get(f"filter_widget_{name}", [])


def persistent_multiselect(label, options, name, **kwargs):
    saved_key = f"saved_filter_{name}"
    widget_key = f"filter_widget_{name}"
    valid_options = list(options)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = [
            value for value in st.session_state.get(saved_key, []) if value in valid_options
        ]
    return st.multiselect(
        label,
        valid_options,
        key=widget_key,
        on_change=_save_filter,
        args=(name,),
        **kwargs,
    )


def reset_filter_button():
    def reset():
        for name in FILTER_NAMES:
            st.session_state[f"saved_filter_{name}"] = []
            st.session_state[f"filter_widget_{name}"] = []

    st.sidebar.button("Reset filters", on_click=reset, use_container_width=True)
