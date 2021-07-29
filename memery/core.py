__all__ = ['index_flow', 'query_flow']

import time

import streamlit.caching
import torch
import gc

from pathlib import Path
from loader import get_image_files, archive_loader, db_loader, treemap_loader, check_for_new_files
from crafter import crafter, preproc
from encoder import image_encoder, text_encoder, image_query_encoder
from indexer import join_all, build_treemap, save_archives, save_encodings
from ranker import ranker, nns_to_files


def index_flow(path, filepaths=None):
    root = Path(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if filepaths is None:
        filepaths = get_image_files(root)
    if len(filepaths) > 100000:
        return chunked_encode(path)
    archive_db, new_files = archive_loader(filepaths, root, 'cpu')
    print(f"Loaded {len(archive_db)} encodings")
    print(f"Encoding {len(new_files)} new images")

    crafted_files = crafter(new_files, device, batch_size=32)
    new_embeddings = image_encoder(crafted_files, device)

    db = join_all(archive_db, new_files, new_embeddings)
    print("Building treemap")
    t = build_treemap(db)

    print(f"Saving {len(db)} encodings")
    save_paths = save_archives(root, t, db)

    return (save_paths)


def chunked_encode(path, maxchunks=-1, filepaths=None, maxnew=100000):
    root = Path(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if filepaths is None:
        filepaths = get_image_files(root)
    archive_db = {}
    new_files = [1]
    nchunks = 0
    while len(new_files) > 0:

        print("Processing chunk ", nchunks)
        archive_db, new_files = archive_loader(filepaths, root, 'cpu', maxnew=maxnew)
        print(f"Loaded {len(archive_db)} encodings")
        print(f"Encoding {len(new_files)} new images")

        #     start_time = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        crafted_files = crafter(new_files, device, batch_size=32)
        new_embeddings = image_encoder(crafted_files, device)

        db = join_all(archive_db, new_files, new_embeddings)

        save_paths = save_encodings(root, db)
        nchunks += 1
        if maxchunks > 0 and nchunks >= maxchunks:
            print("Finishing encoding due to reaching chunk limit")
            break

    print("Building treemap")
    t = build_treemap(db)

    print(f"Saving treemap with {len(db)} encodings")
    save_paths = save_archives(root, t, db)

    return (save_paths)


def query_flow(path, query=None, image_query=None, filepaths=None):
    start_time = time.time()
    print('starting timer')
    root = Path(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Checking files")
    dbpath = root / 'memery.pt'
    db = db_loader(dbpath, 'cpu')
    treepath = root / 'memery.ann'
    treemap = treemap_loader(treepath)
    if filepaths is None:
        filepaths = get_image_files(root)
    if treemap == None or len(db) != len(filepaths):
        print("checking for new files", treemap is None, len(db), len(filepaths))
        if check_for_new_files(filepaths, db):
            print('Indexing')
            dbpath, treepath = index_flow(root)
            streamlit.caching.clear_cache()
            treemap = treemap_loader(Path(treepath))
            db = db_loader(dbpath, 'cpu')

    print('Converting query')
    if image_query:
        img = preproc(image_query)
    if query and image_query:
        text_vec = text_encoder(query, device)
        image_vec = image_query_encoder(img, device)
        query_vec = text_vec + image_vec
    elif query:
        query_vec = text_encoder(query, device)
    elif image_query:
        query_vec = image_query_encoder(img, device)
    else:
        print('No query!')

    print(f"Searching {len(db)} images")
    indexes = ranker(query_vec, treemap)
    ranked_files = nns_to_files(db, indexes)

    print(f"Done in {time.time() - start_time} seconds")

    return ranked_files
