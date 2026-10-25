#!/usr/bin/env python3
"""Check the records package against MANIFEST.json, the seals and SCRUB_LOG.json.

Run from the repository root:

    python3 integrity/verify.py

The script uses the Python standard library only. It reads files. It writes nothing.

Checks:
  1. Each file in MANIFEST.json exists, and its SHA-256 and size agree with the manifest.
  2. Each copy that is not scrubbed and not redacted has the SHA-256 of the original.
  3. The original SHA-256 agrees with the recovery manifest line and with the seal of the folder.
  4. Each seal file in records/ lists, for each file that it names, the original SHA-256 in MANIFEST.json.
  5. Each path replacement and each redaction in SCRUB_LOG.json is at the recorded position in the copy.
  6. No file in records/ is absent from MANIFEST.json.

With --home PREFIX (version 1.0 files) and --prefix NAME=VALUE (version 1.1 files, for example
--prefix SNAPSHOT=...), the script also rebuilds each scrubbed original from the copy and the log,
and compares its SHA-256 with the original SHA-256. Only the owner knows the values. A file whose
rules need a value that is not given is not rebuilt.

A redacted file cannot be rebuilt, because the removed text is not in this repository.
The script lists the redacted files separately and does not rebuild them.

Version 1.1 seal files list the file hashes under "sha256". Version 1.0 seal files list them under "files".
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return f.read()


def placeholders(shown):
    """The withheld names in a rule, for example <HOME> or <SNAPSHOT>."""
    names, i = [], shown.find("<")
    while i >= 0:
        j = shown.find(">", i)
        names.append(shown[i + 1:j])
        i = shown.find("<", j)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", help="the withheld home folder prefix of version 1.0 (owner only)")
    ap.add_argument("--prefix", action="append", default=[], metavar="NAME=VALUE",
                    help="a withheld value of version 1.1, for example SNAPSHOT=... (owner only)")
    args = ap.parse_args()
    values = dict(p.split("=", 1) for p in args.prefix)
    if args.home:
        values["HOME"] = args.home
    manifest = json.loads(read("MANIFEST.json"))
    scrub = json.loads(read("SCRUB_LOG.json"))
    errors = []
    redactions = scrub.get("redactions", [])
    redacted = {r["new_path"] for r in redactions}
    by_new = {f["new_path"]: f for f in manifest["files"]}
    by_orig = {f["original_path"]: f for f in manifest["files"] + manifest["excluded"]}

    # 1, 2 and 3
    for f in manifest["files"]:
        data = read(f["new_path"])
        if sha256(data) != f["sha256_copy"] or len(data) != f["bytes_copy"]:
            errors.append("copy hash or size differs: " + f["new_path"])
        if bool(f.get("redacted")) != (f["new_path"] in redacted):
            errors.append("redacted flag disagrees with SCRUB_LOG.json: " + f["new_path"])
        if not f["scrubbed"] and not f.get("redacted") and f["sha256_copy"] != f["sha256_original"]:
            errors.append("unchanged copy differs from original: " + f["new_path"])
    for f in manifest["files"] + manifest["excluded"]:
        for key in ("sha256_recovery_manifest", "sha256_in_seal"):
            if f[key] and f[key] != f["sha256_original"]:
                errors.append(key + " differs: " + f["original_path"])

    # 3, against the recovery manifest lines themselves
    n_lines = 0
    for line in read(manifest["recovery_manifest_lines"]["file"]).decode("utf-8").splitlines():
        d = json.loads(line)
        n_lines += 1
        f = by_orig.get(d["destination"])
        if f is None:
            errors.append("recovery line for a file that MANIFEST does not list: " + d["destination"])
        elif d["sha256"] != f["sha256_original"]:
            errors.append("recovery line hash differs: " + d["destination"])

    # 4
    n_seal = 0
    for seal in manifest["seals"]:
        folder = os.path.dirname(seal)
        data = json.loads(read(seal))
        listed = data.get("files")
        if not isinstance(listed, dict):
            listed = data.get("sha256", {})
        for name, digest in listed.items():
            orig = "ember/" + os.path.relpath(os.path.join(folder, name), "records")
            f = by_orig.get(orig)
            if f is None:
                continue  # the seal names a file that is outside the folders in scope
            n_seal += 1
            if f["sha256_original"] != digest:
                errors.append("seal hash differs: " + seal + " -> " + name)

    # 5, redactions
    n_red = 0
    for r in redactions:
        f = by_new.get(r["new_path"])
        if f is None or f["sha256_copy"] != r["sha256_copy"] or f["sha256_original"] != r["sha256_original"]:
            errors.append("redaction hashes disagree with MANIFEST.json: " + r["new_path"])
        lines = read(r["new_path"]).decode("utf-8").split("\n")
        for e in r["redactions"]:
            n_red += 1
            c = e["col_copy"] - 1
            if lines[e["line"] - 1][c:c + len(e["replacement"])] != e["replacement"]:
                errors.append("redaction not at the logged position: %s line %d" % (r["new_path"], e["line"]))

    # 5, path replacements, and the rebuild with --home
    rule_old = {r["rule"]: r["original_shown"] for r in scrub["rules"]}
    n_rep = n_rebuilt = n_rebuild_target = 0
    for s in scrub["files"]:
        lines = read(s["new_path"]).decode("utf-8").split("\n")
        by_line = {}
        for e in s["replacements"]:
            n_rep += 1
            text = lines[e["line"] - 1]
            c = e["col_copy"] - 1
            if text[c:c + len(e["replacement"])] != e["replacement"]:
                errors.append("replacement not at the logged position: %s line %d" % (s["new_path"], e["line"]))
            by_line.setdefault(e["line"] - 1, []).append(e)
        needed = {n for e in s["replacements"] for n in placeholders(rule_old[e["rule"]])}
        if values and s["new_path"] not in redacted and needed <= set(values):
            n_rebuild_target += 1
            for ln, es in by_line.items():
                t = lines[ln]
                for e in sorted(es, key=lambda e: -e["col_copy"]):
                    c = e["col_copy"] - 1
                    old = rule_old[e["rule"]]
                    for n in placeholders(old):
                        old = old.replace("<" + n + ">", values[n])
                    t = t[:c] + old + t[c + len(e["replacement"]):]
                lines[ln] = t
            if sha256("\n".join(lines).encode("utf-8")) != s["sha256_original"]:
                errors.append("rebuilt original differs: " + s["new_path"])
            else:
                n_rebuilt += 1

    # 6
    for dp, dn, fn in os.walk(os.path.join(ROOT, "records")):
        for name in fn:
            rel = os.path.relpath(os.path.join(dp, name), ROOT)
            if rel not in by_new:
                errors.append("file not in MANIFEST.json: " + rel)

    print("files checked:            %d" % len(manifest["files"]))
    print("excluded files listed:    %d" % len(manifest["excluded"]))
    print("recovery lines checked:   %d" % n_lines)
    print("seal entries checked:     %d" % n_seal)
    print("scrub entries checked:    %d" % n_rep)
    print("redactions checked:       %d" % n_red)
    if values:
        print("originals rebuilt:        %d of %d scrubbed files that are not redacted" % (n_rebuilt, n_rebuild_target))
    print("redacted files (not rebuilt; the removed text is not in this repository): %d" % len(redacted))
    for path in sorted(redacted):
        print("  " + path)
    if errors:
        print("FAIL: %d problems" % len(errors))
        for e in errors[:50]:
            print("  " + e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
