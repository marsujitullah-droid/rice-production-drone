# === Pembersihan label segmentasi hasil Auto Label (aturan sama untuk train/valid/test) ===
import cv2, numpy as np, pandas as pd, shutil
from pathlib import Path

S = 640                      # ukuran gambar dataset (Fit 640x640)
MIN_AREA = 0.002             # poligon < 0,2% luas gambar dianggap potongan/noise
DUP_IOU = 0.70               # dua poligon dengan IoU > 0,70 dianggap duplikat -> simpan yang lebih besar
CONTAIN = 0.80               # poligon yang >80% luasnya berada di dalam poligon lain dianggap bersarang

def read_polys(lf):
    out = []
    for line in Path(lf).read_text().splitlines():
        v = line.split()
        if len(v) >= 7:
            out.append((v[0], np.array(v[1:], float).reshape(-1, 2)))
    return out

def to_mask(xy):
    m = np.zeros((S, S), np.uint8)
    cv2.fillPoly(m, [np.round(xy * S).astype(np.int32)], 1)
    return m.astype(bool)

def clean(polys):
    masks = [to_mask(xy) for _, xy in polys]
    area = np.array([m.sum() for m in masks], float)
    keep = area >= MIN_AREA * S * S
    n_kecil = int((~keep).sum())
    idx = [i for i in range(len(polys)) if keep[i]]
    # duplikat: IoU tinggi -> buang yang lebih kecil
    n_dup = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if not (keep[i] and keep[j]): continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            if inter / (area[i] + area[j] - inter) > DUP_IOU:
                keep[j if area[j] < area[i] else i] = False; n_dup += 1
    # bersarang: poligon besar yang memuat >=2 poligon lain = "wadah" salah -> buang wadahnya;
    # jika hanya memuat 1 poligon, buang yang kecil (potongan di dalam petak)
    idx = [i for i in range(len(polys)) if keep[i]]
    isi = {i: [j for j in idx if j != i and np.logical_and(masks[i], masks[j]).sum() > CONTAIN * area[j]
               and area[j] < area[i]] for i in idx}
    n_wadah = n_dalam = 0
    for i in sorted(idx, key=lambda k: -area[k]):
        if not keep[i]: continue
        anak = [j for j in isi[i] if keep[j]]
        if len(anak) >= 2: keep[i] = False; n_wadah += 1
        elif len(anak) == 1: keep[anak[0]] = False; n_dalam += 1
    return [p for p, k in zip(polys, keep) if k], dict(kecil=n_kecil, duplikat=n_dup, wadah=n_wadah, bersarang=n_dalam)

def clean_dataset(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    rows = []
    for lf in sorted(dst.glob('*/labels/*.txt')):
        polys = read_polys(lf)
        kept, info = clean(polys)
        lf.write_text('\n'.join(c + ' ' + ' '.join(f'{v:.6f}' for v in xy.ravel()) for c, xy in kept) + ('\n' if kept else ''))
        rows.append(dict(split=lf.parent.parent.name, foto=lf.stem.split('.rf.')[0], sebelum=len(polys), sesudah=len(kept), **info))
    return pd.DataFrame(rows).sort_values('sebelum', ascending=False)

def overlay(img_path, polys, color):
    im = cv2.imread(str(img_path)); h, w = im.shape[:2]
    for _, xy in polys:
        cv2.polylines(im, [np.round(xy * [w, h]).astype(np.int32)], True, color, 2)
    return im
