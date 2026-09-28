"""List and download from a public S3 bucket using only the standard library.

Fallback for when the AWS CLI is not installed. The SEA-AD buckets are public,
so anonymous HTTPS against the REST API works and no credentials are involved.

  python scripts/s3_public.py list sea-ad-single-cell-profiling --suffix .h5ad
  python scripts/s3_public.py get  sea-ad-single-cell-profiling <key> data/raw/
"""
import sys, os, urllib.request, urllib.parse, xml.etree.ElementTree as ET

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def endpoint(bucket, region="us-west-2"):
    return f"https://{bucket}.s3.{region}.amazonaws.com/"


def list_objects(bucket, region="us-west-2", prefix="", suffix=None):
    token, out = None, []
    while True:
        q = {"list-type": "2", "max-keys": "1000"}
        if prefix:
            q["prefix"] = prefix
        if token:
            q["continuation-token"] = token
        url = endpoint(bucket, region) + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{NS}Contents"):
            key = c.findtext(f"{NS}Key", "")
            size = int(c.findtext(f"{NS}Size", "0"))
            if suffix and not key.endswith(suffix):
                continue
            out.append((key, size))
        if root.findtext(f"{NS}IsTruncated", "false") != "true":
            break
        token = root.findtext(f"{NS}NextContinuationToken")
        if not token:
            break
    return out


def download(bucket, key, dest_dir, region="us-west-2"):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(key))
    url = endpoint(bucket, region) + urllib.parse.quote(key)
    print(f"GET {url}\n -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done/2**30:6.2f} / {total/2**30:.2f} GB  ({pct:5.1f}%)",
                          end="", flush=True)
    print(f"\ndone: {dest}")
    return dest


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    cmd, bucket = argv[1], argv[2]
    region = os.environ.get("AWS_REGION", "us-west-2")
    if cmd == "list":
        suffix = None
        if "--suffix" in argv:
            suffix = argv[argv.index("--suffix") + 1]
        prefix = ""
        if "--prefix" in argv:
            prefix = argv[argv.index("--prefix") + 1]
        rows = list_objects(bucket, region, prefix, suffix)
        for key, size in sorted(rows):
            print(f"{size/2**30:10.2f} GB  {key}")
        print(f"\n{len(rows)} object(s)")
    elif cmd == "get":
        download(bucket, argv[3], argv[4] if len(argv) > 4 else ".", region)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
