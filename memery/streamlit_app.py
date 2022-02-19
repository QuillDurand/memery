__all__ = ['st_redirect', 'st_stdout', 'st_stderr', 'send_image_query', 'send_text_query', 'path', 'text_query',
           'image_query', 'im_display_zone', 'logbox', 'sizes']


import streamlit as st
import core

from pathlib import Path
from PIL import Image

from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from contextlib import contextmanager
from io import StringIO
import sys
import time
import copy


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)
        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


@st.cache(allow_output_mutation=True, max_entries=10)
def send_image_query(path, text_query, image_query, filepaths=None):
    ranked = core.query_flow(path, text_query, image_query=image_query, filepaths=filepaths)
    return(ranked)


@st.cache(allow_output_mutation=True, max_entries=10)
def send_text_query(path, text_query, filepaths=None):
    ranked = core.query_flow(path, text_query, filepaths=filepaths)
    return(ranked)


st.sidebar.title("Memery")

path = st.sidebar.text_input(label='Directory', value='')
text_query = st.sidebar.text_input(label='Text query', value='')
image_query = st.sidebar.file_uploader(label='Image query')
im_display_zone = st.sidebar.beta_container()
logbox = st.sidebar.beta_container()


sizes = {'small': 115, 'medium':230, 'large':332, 'xlarge':600}

l, m, r = st.beta_columns([4,1,1])
with l:
    num_images = st.slider(label='Number of images',value=12)
with m:
    size_choice = st.selectbox(label='Image width', options=[k for k in sizes.keys()], index=1)
with r:
    captions_on = st.checkbox(label="Caption filenames", value=False)


@st.cache(allow_output_mutation=True, show_spinner=False, max_entries=1000)
def load_image(path):
    return Image.open(path).convert('RGB')


if text_query or image_query:

    with logbox:
        with st_stdout('info'):
            if image_query is not None:
                img = load_image(image_query)
                with im_display_zone:
                    st.image(img)
                ranked = send_image_query(path, text_query, img)
            else:
                ranked = send_text_query(path, text_query)
    start_time = time.time()
    ranked = copy.copy(ranked)
    ims = [load_image(o) for o in ranked[:num_images]]
    names = [o.replace(path, '') for o in ranked[:num_images]]
    print(f"copied rankings and loaded images in {time.time() - start_time} seconds")
    start_time = time.time()
    if captions_on:
        images = st.image(ims, width=sizes[size_choice], channels='RGB', caption=names)
    else:
        images = st.image(ims, width=sizes[size_choice], channels='RGB')
    print(f"Displayed images in {time.time() - start_time} seconds")
