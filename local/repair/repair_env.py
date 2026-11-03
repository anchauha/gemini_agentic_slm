#!/usr/bin/env python3
"""Per-task environment repair, run inside a sandbox container after the harness's own setup.

Makes the public tasks gradable the way the scorer-like sandbox of Kaggle discussion 751487
(Zbigniew Kirchner, gold patches pass on 127/129) does, adapted to the Docker backend:

1. Test-only packages missing from the competition wheels/ (baked into /opt/repair/wheels), with
   pytest 8.4.2 and pytest-timeout 2.1.0 as reported for the scorer image. Packages that break
   other repositories' tests (brotli, httpbin, ...) go only to the repository that needs them.
2. The repository's own released package, which the harness injects from wheels/, is removed so
   it cannot shadow the code under test in /workspace.
3. For every dependency in the snapshot's pyproject.toml (and what those wheels require), the
   newest version the snapshot allows. The harness always injects the newest wheel, so old
   fastapi commits otherwise meet starlette 1.6.
4. A small per-task calibration found by reading failing gold-patch output (see CALIBRATION).

Usage (inside the container): python3 /opt/repair/repair_env.py <owner/repo> <instance_id>
"""

import email.parser
import importlib.metadata as md
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import InvalidVersion, Version

REPAIR = Path("/opt/repair")
WHEEL_DIRS = [REPAIR / "competition", REPAIR / "wheels"]
SITE = Path("/usr/local/lib/python3.13/site-packages")
WORKSPACE = Path("/workspace")
MARKER = SITE / ".repair_base_done"

OWN = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
ONLY_FOR = {"pytest_httpbin": {"psf/requests"}, "httpbin": {"psf/requests"}, "flasgger": {"psf/requests"},
            "brotlicffi": {"psf/requests", "encode/httpx"}, "brotli": {"encode/httpx"},
            "zstandard": {"encode/httpx"}}
PINNED = {"pytest": "8.4.2", "pytest_timeout": "2.1.0"}
CALIBRATION = {  # task -> exact versions, from the gold patch's pytest output
    # old rich tests compare colored output 1:1, so they need the Pygments of their time
    "rich_3064": {"pygments": "2.15.1"}, "rich_3130": {"pygments": "2.15.1"}, "rich_3296": {"pygments": "2.16.1"},
    "rich_3469": {"pygments": "2.18.0"}, "rich_3782": {"pygments": "2.19.2"}, "rich_4070": {"pygments": "2.19.2"},
    "rich_4079": {"pygments": "2.20.0"},
    # the httpbin test server (werkzeug 3.1) crashes on a redirect to port -1, werkzeug 2.3 does not
    "requests_6589": {"werkzeug": "2.3.8", "flask": "2.3.3"}, "requests_6592": {"werkzeug": "2.3.8", "flask": "2.3.3"},
    "requests_6629": {"werkzeug": "2.3.8", "flask": "2.3.3"},
}
ASYNC_SHIM = {"fastapi_15745", "fastapi_15763", "fastapi_15785"}  # async tests, and the harness passes -p no:anyio

norm = lambda n: re.sub(r"[-_.]+", "_", n).lower()


def fits(wheel: Path) -> bool:
    py, abi, plat = wheel.name[:-4].split("-")[-3:]
    if plat != "any" and "linux" not in plat:
        return False
    if abi == "abi3":  # cp39-abi3 works on 3.9 and newer
        return int(py.split(".")[0][2:]) <= 313
    return any(t.startswith("py3") for t in py.split(".")) or "cp313" in py.split(".")


def index() -> dict[str, list[tuple[Version, Path]]]:
    wheels: dict[str, list[tuple[Version, Path]]] = {}
    for d in WHEEL_DIRS:
        for w in d.glob("*.whl"):
            name, ver = w.name.split("-")[:2]
            try:
                wheels.setdefault(norm(name), []).append((Version(ver), w))
            except InvalidVersion:
                continue
    return {k: [o for o in v if fits(o[1])] for k, v in wheels.items()}


def installed() -> dict[str, str]:
    return {norm(d.metadata["Name"]): d.version for d in md.distributions(path=[str(SITE)]) if d.metadata["Name"]}


def purge(name: str, tops: set[str] = frozenset()) -> None:
    """Deletes an installed distribution. The harness unpacks wheels without a RECORD, so pip cannot."""
    for info in SITE.glob("*.dist-info"):
        if norm(info.name.rsplit("-", 1)[0]) != name:
            continue
        tl = info / "top_level.txt"
        tops = set(tops) | (set(tl.read_text().split()) if tl.exists() else set())
        shutil.rmtree(info, ignore_errors=True)
    for top in tops:
        for p in (SITE / top, SITE / f"{top}.py"):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


def pip_install(wheels: list[Path]) -> None:
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            tops = {n.split("/", 1)[0] for n in z.namelist()}
        purge(norm(w.name.split("-")[0]), {t.removesuffix(".py") for t in tops if not t.endswith((".dist-info", ".data"))})
    if wheels:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--ignore-installed",
                        "--ignore-requires-python", "--disable-pip-version-check", "--root-user-action=ignore",
                        *map(str, wheels)], check=True)


def remove_released_own(own: str) -> None:
    """Drops the repo's own wheel that the harness injected; /workspace provides the package."""
    if (SITE / own).is_dir():
        purge(own, {own})


def snapshot_requirements() -> list[str]:
    try:
        data = tomllib.loads((WORKSPACE / "pyproject.toml").read_text())
    except Exception:
        return []
    reqs = list(data.get("project", {}).get("dependencies", []))
    for name, spec in data.get("tool", {}).get("poetry", {}).get("dependencies", {}).items():
        if name == "python":
            continue
        spec = spec if isinstance(spec, str) else spec.get("version", "")
        m = re.fullmatch(r"\^(\d+)(\.\d+)?(\.\d+)?", spec.strip())
        spec = f">={spec.strip()[1:]},<{int(m.group(1)) + 1}" if m else ("" if spec.strip() == "*" else spec)
        reqs.append(name + spec)
    return reqs


def allowed_versions(wheels, repo: str, own: str) -> dict[str, Path]:
    """Newest wheel the snapshot allows for each dependency, followed through Requires-Dist."""
    picks, queue, seen = {}, snapshot_requirements(), set()
    while queue:
        try:
            req = Requirement(queue.pop(0))
        except Exception:
            continue
        key = norm(req.name)
        if key in seen or key == own or (req.marker and not req.marker.evaluate({"extra": ""})):
            continue
        seen.add(key)
        options = [(v, w) for v, w in wheels.get(key, []) if req.specifier.contains(v, prereleases=True)
                   and not (key in ONLY_FOR and repo not in ONLY_FOR[key])]
        if not options:
            continue
        best = max(options)[1]
        picks[key] = best
        with zipfile.ZipFile(best) as z:
            meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
            msg = email.parser.Parser().parsestr(z.read(meta).decode())
            queue += msg.get_all("Requires-Dist") or []
    return picks


def main(repo: str, task_id: str) -> None:
    own = norm(OWN.get(repo, ""))
    wheels = index()
    competition = {norm(w.name.split("-")[0]) for w in (REPAIR / "competition").glob("*.whl")}

    if not MARKER.exists():  # once per container: missing test packages, scorer pytest, per-repo extras
        base = []
        for name, opts in wheels.items():
            if not opts or name == own or name == "pygments":
                continue
            if name in PINNED:
                base += [w for v, w in opts if str(v) == PINNED[name]][:1]
            elif name in ONLY_FOR:
                if repo in ONLY_FOR[name]:
                    base.append(max(opts)[1])
            elif name not in competition:  # the harness already injected the newest competition wheel
                base.append(max(opts)[1])
        pip_install(base)
        for name in ONLY_FOR:
            if repo not in ONLY_FOR[name] and name in installed():
                purge(name)
        # inline_snapshot's pytest plugin imports packages from site-packages before collection
        for ep in SITE.glob("inline_snapshot-*.dist-info/entry_points.txt"):
            ep.write_text(re.sub(r"\[pytest11\][^\[]*", "", ep.read_text()))
        MARKER.write_text(f"{repo} {task_id}\n")

    remove_released_own(own)
    (SITE / "00_workspace_src.pth").write_text(f"{WORKSPACE}/src\n" if (WORKSPACE / "src").is_dir() else "")

    have = installed()
    picks = {k: w for k, w in allowed_versions(wheels, repo, own).items() if have.get(k) != w.name.split("-")[1]}
    for name, ver in CALIBRATION.get(task_id, {}).items():
        picks[name] = next(w for v, w in wheels[name] if str(v) == ver)
    pip_install(list(picks.values()))
    if task_id in ASYNC_SHIM:
        (SITE / "02_async_shim.pth").write_text(f"{REPAIR / 'shims'}\n")
    print("repair:", task_id, " ".join(sorted(p.name.split("-")[0] + "==" + p.name.split("-")[1] for p in picks.values())))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
