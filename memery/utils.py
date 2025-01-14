__all__ = ['top_bottom', 'copy_topn_images']


def top_bottom(query):
    results = predict_from_text(image_names, image_features, query)
    inv_results = sorted(results, key=lambda o: o[1])
    print(query.upper())
    n = 10
    w = 200
    print(f'top {n}')
    printi([file for file, score in results], n, w)
    print(f'bottom {n}')
    printi([file for file, score in inv_results], n, w)


def copy_topn_images(results, outpath, n):
    for file, score in results[:n]:
        prefix = str(int(10*float(score)))
        filepath = Path(file)
        filename = '-'.join(filepath.parts[-2:])
        outfile = outpath/f'{prefix}-{filename}'
        outfile.write_bytes(filepath.read_bytes())