__all__ = ['join_all', 'build_treemap', 'save_archives', 'save_encodings']


def join_all(db, new_files, new_embeddings):
    start = len(db)
    for i, file in enumerate(new_files):
        path, slug = file
        index = i + start
        db[index] = {
            'slug': slug,
            'fpath': path,
            'embed': new_embeddings[i],
        }
    return db


from annoy import AnnoyIndex


def build_treemap(db):
    treemap = AnnoyIndex(len(db.items().__iter__().__next__()[1]['embed']), 'angular')
    for k, v in db.items():
        treemap.add_item(k, v['embed'])

    # Build the treemap, with 5 trees rn
    treemap.build(5, n_jobs=5)

    return treemap


import torch


def save_archives(root, treemap, db):
    dbpath = root/'memery.pt'
    if dbpath.exists():
        dbpath.unlink()
    torch.save(db, dbpath)

    treepath = root/'memery.ann'
    if treepath.exists():
        treepath.unlink()
    treemap.save(str(treepath))

    return str(dbpath), str(treepath)


def save_encodings(root, db):
    dbpath = root/'memery.pt'
    if dbpath.exists():
        dbpath.unlink()
    torch.save(db, dbpath)
    return str(dbpath)
