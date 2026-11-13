"""Read members of the official UCF_Crimes.zip via HTTP Range requests (curl), no full download needed.

curl is used instead of urllib because the CRCV server's TLS chain only validates with macOS curl/keychain.
"""
import io, subprocess, sys, zipfile

URL = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
SIZE = 102_957_372_377  # Content-Length from HEAD, 2026-11-03


def curl_range(start: int, end: int, out=None) -> bytes:
    cmd = ["curl", "-sSf", "--retry", "5", "--retry-all-errors", "-r", f"{start}-{end}", URL]
    if out is not None:
        cmd += ["-o", out]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


class HttpFile(io.RawIOBase):
    def __init__(self):
        self.size, self.pos = SIZE, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = curl_range(self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip():
    return zipfile.ZipFile(io.BufferedReader(HttpFile(), buffer_size=4 << 20))


if __name__ == "__main__":
    z = open_remote_zip()
    print("filename\tfile_size\tcompress_size\tcompress_type\theader_offset\tcrc32")
    for i in z.infolist():
        print(f"{i.filename}\t{i.file_size}\t{i.compress_size}\t{i.compress_type}\t{i.header_offset}\t{i.CRC:08x}")
