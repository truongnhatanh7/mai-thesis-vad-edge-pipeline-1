"""Extract selected members of the official UCF_Crimes.zip by HTTP range (curl), verifying CRC32 + size.

usage: extract_members.py <split_list.txt> <out_root> <log.tsv> [workers]
Each list line is a path relative to UCF_Crimes/Videos/ (format of Anomaly_Test.txt).
"""
import hashlib, os, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from http_zip import curl_range

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTING = pd.read_csv(os.path.join(HERE, "raw/UCF_Crimes.zip.listing.tsv"), sep="\t").set_index("filename")


def extract(rel, out_root):
    name = f"UCF_Crimes/Videos/{rel}"
    row = LISTING.loc[name]
    dst = os.path.join(out_root, rel)
    if os.path.exists(dst) and os.path.getsize(dst) == row.file_size:
        return rel, "skip-exists", row.file_size, ""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    off = int(row.header_offset)
    hdr = curl_range(off, off + 29)
    sig, *_ = struct.unpack("<I", hdr[:4])
    assert sig == 0x04034B50, f"bad local header {rel}"
    fn_len, extra_len = struct.unpack("<HH", hdr[26:30])
    start = off + 30 + fn_len + extra_len
    part = dst + ".deflate.part"
    curl_range(start, start + int(row.compress_size) - 1, out=part)
    assert os.path.getsize(part) == row.compress_size, f"short read {rel}"
    d = zlib.decompressobj(-15)
    crc, n, h = 0, 0, hashlib.sha256()
    with open(part, "rb") as fi, open(dst + ".part", "wb") as fo:
        while chunk := fi.read(8 << 20):
            out = d.decompress(chunk)
            crc = zlib.crc32(out, crc); n += len(out); h.update(out); fo.write(out)
        out = d.flush()
        crc = zlib.crc32(out, crc); n += len(out); h.update(out); fo.write(out)
    assert n == row.file_size and f"{crc:08x}" == row.crc32, f"CRC/size mismatch {rel}"
    os.replace(dst + ".part", dst)
    os.remove(part)
    return rel, "ok", n, h.hexdigest()


def main():
    lst, out_root, log = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    rels = [l.strip() for l in open(lst) if l.strip()]
    new = not os.path.exists(log)
    with open(log, "a") as lf, ThreadPoolExecutor(workers) as ex:
        if new:
            lf.write("utc\tvideo\tstatus\tbytes\tsha256\n")
        for fut in [ex.submit(extract, r, out_root) for r in rels]:
            try:
                rel, st, n, sha = fut.result()
            except Exception as e:  # keep going; failures are logged and re-run later
                rel, st, n, sha = "?", f"FAIL {e}", 0, ""
            lf.write(f"{datetime.now(timezone.utc):%FT%TZ}\t{rel}\t{st}\t{n}\t{sha}\n"); lf.flush()


if __name__ == "__main__":
    main()
