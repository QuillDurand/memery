__all__ = ['slugify', 'verify_image', 'get_image_files', 'device', 'archive_loader', 'db_loader', 'treemap_loader',
           'check_for_new_files']

from pathlib import Path

import streamlit
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process, Queue
import time
import os
import streamlit as st


def slugify(filepath):
    return f'{filepath.stem}_{str(filepath.stat().st_mtime).split(".")[0]}'


def verify_image(f, full_verify=True):
    try:
        img = Image.open(f)  # open the image file
        if full_verify:
            img.verify()  # verify that it is, in fact an image
        return True
    except Exception as e:
        try:
            print(f'Deleting bad file: {f}\ndue to {type(e)}')
            os.remove(f)
        except:
            pass


def verification_worker(inqueue, outqueue, full_verify=True):
    try:
        while True:
            path, slug = inqueue.get_nowait()
            try:
                img = Image.open(path)  # open the image file
                if full_verify:
                    img.verify()  # verify that it is, in fact an image
                outqueue.put((str(path), slug))
            except Exception as e:
                print(f'Deleting bad file: {path}\ndue to {type(e)}')
                os.remove(path)
                pass
    except:
        outqueue.put("done")


@st.cache(allow_output_mutation=True)
def get_image_files(path):
    print(f"reading filepaths for {path} from disk")
    img_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    return [(f, slugify(f)) for f in tqdm(path.rglob('*')) if f.suffix in img_extensions]


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_multiprocessing(filepaths, archive_slugs, nworkers=20, full_verify=True):
    starttime = time.time()
    taskqueue = Queue()
    resultqueue = Queue()
    [taskqueue.put((path, slug)) for path, slug in tqdm(filepaths) if slug not in archive_slugs]
    workers = [Process(target=verification_worker, args=(taskqueue, resultqueue, full_verify)) for i in range(nworkers)]
    [w.start() for w in workers]
    result = []
    workersdone = 0
    while workersdone < nworkers:
        res = resultqueue.get()
        if res == 'done':
            workersdone += 1
            continue
        result.append(res)
        if (len(result) % 1000) == 0:
            print(f"Finished {len(result)}/{len(filepaths)} in {int(time.time() - starttime)} seconds")
    return result


def archive_loader(filepaths, root, device, maxnew=-1):
    dbpath = root / 'memery.pt'
    #     dbpath_backup = root/'memery.pt'
    streamlit.caching.clear_cache()
    db = db_loader(dbpath, device)
    print("fetching slugs")
    current_slugs = set([slug for path, slug in tqdm(filepaths)])
    print("finding already archived")
    archive_db = {i: db[item[0]] for i, item in tqdm(enumerate(db.items())) if item[1]['slug'] in current_slugs}
    archive_slugs = set([v['slug'] for v in tqdm(archive_db.values())])
    print("verifying files")
    #new_files = [(str(path), slug) for path, slug in tqdm(filepaths) if slug not in archive_slugs and verify_image(path, full_verify=False)]
    # new_files = verify_multiprocessing(filepaths, archive_slugs, full_verify=False)
    new_files = [(str(path), slug) for path, slug in tqdm(filepaths) if slug not in archive_slugs]
    if maxnew > 0:
        new_files = new_files[:maxnew]

    return archive_db, new_files


def check_for_new_files(filepaths, db):
    print("fetching slugs")
    current_slugs = set([slug for path, slug in tqdm(filepaths)])
    print("finding already archived")
    archive_db = {i: db[item[0]] for i, item in tqdm(enumerate(db.items())) if item[1]['slug'] in current_slugs}
    archive_slugs = set([v['slug'] for v in tqdm(archive_db.values())])
    new_files = [(str(path), slug) for path, slug in tqdm(filepaths) if slug not in archive_slugs]
    if len(new_files) > 0:
        return True
    return False


@st.cache(allow_output_mutation=True, max_entries=1)
def db_loader(dbpath, device):
    # check for savefile or backup and extract
    if Path(dbpath).exists():
        db = torch.load(dbpath, device)
    else:
        db = {}
    return db


from annoy import AnnoyIndex


@st.cache(allow_output_mutation=True, max_entries=1)
def treemap_loader(treepath):
    treemap = AnnoyIndex(768, 'angular')

    if treepath.exists():
        treemap.load(str(treepath))
    else:
        treemap = None
    return treemap
