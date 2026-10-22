"""Where the raw dataset files live, and fetching the pinned ones.

Each entry of TRAIN_SOURCES / BENCH_SOURCES gives how to fetch a dataset
(kind "hf": a Hugging Face dataset repo, kind "url": a plain download, GitHub
raw URLs pinned by commit) and its revision.

Files live outside the workspace:
  - kind "hf" datasets are in the Hugging Face cache (HF_HOME), fetched with
    snapshot_download at the pinned revision; `root` returns the snapshot
    folder.
  - kind "url" datasets, and the ones added by hand (contractnli, ...), are in
    <HF_HOME>/llm_jev/<name>/ (override with the LLM_JEV_DATA variable).
The sha256 of every fetched file is written to <data home>/<name>/SOURCES.json.
"""

import functools
import hashlib
import json
import os
import pathlib
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any

from huggingface_hub import constants
from loguru import logger

TRAIN_SOURCES: dict[str, dict[str, Any]] = {
    "clinc150": {
        "kind": "hf", "repo": "clinc/clinc_oos",
        "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
        "files": ["plus/train-00000-of-00001.parquet",
                  "plus/validation-00000-of-00001.parquet", "README.md"],
    },
    "massive": {
        "kind": "url", "revision": "MASSIVE 1.1 release tarball",
        "files": {"amazon-massive-dataset-1.1.tar.gz":
                  "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
                  "amazon-massive-dataset-1.1.tar.gz"},
    },
    "bitext": {
        "kind": "hf",
        "repo": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "revision": "430d1a89bd93bd1fa23c16f29dd53e73f0087443",
        "files": ["Bitext_Sample_Customer_Support_Training_Dataset_27K_"
                  "responses-v11.csv", "README.md"],
    },
    "reclor": {  # the official zip is behind a form; a mirror of it
        "kind": "hf", "repo": "metaeval/reclor",
        "revision": "0414b39bfc39b2d555b48fa245fc198744c46215",
        "files": ["train.json", "val.json", "README.md"],
    },
    "logiqa2": {
        "kind": "url", "revision": "csitfun/LogiQA2.0@955e1d3d",
        "files": {
            f"{s}.txt": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/"
            f"955e1d3df6c59d9bfb44d9913da1e1a27ec14e18/logiqa/DATA/LOGIQA/{s}.txt"
            for s in ("train", "dev")
        },
    },
    "cosmosqa": {
        "kind": "url", "revision": "wilburOne/cosmosqa@b6eb99cc",
        "files": {
            f"{s}.csv": "https://raw.githubusercontent.com/wilburOne/cosmosqa/"
            f"b6eb99cca4e2a51dd28a9a6f562534872d851639/data/{s}.csv"
            for s in ("train", "valid")
        },
    },
    "piqa": {
        "kind": "url", "revision": "yonatanbisk.com/piqa/data (unversioned)",
        "files": {
            f: f"https://yonatanbisk.com/piqa/data/{f}"
            for f in ("train.jsonl", "train-labels.lst", "valid.jsonl",
                      "valid-labels.lst")
        },
    },
    "anli": {
        "kind": "url", "revision": "AI2 alphanli-train-dev.zip (unversioned)",
        "files": {"alphanli-train-dev.zip":
                  "https://storage.googleapis.com/ai2-mosaic/public/alphanli/"
                  "alphanli-train-dev.zip"},
    },
    "proofwriter": {
        "kind": "url", "revision": "proofwriter-dataset-V2020.12.3",
        "files": {"proofwriter-dataset-V2020.12.3.zip":
                  "https://aristo-data-public.s3.amazonaws.com/proofwriter/"
                  "proofwriter-dataset-V2020.12.3.zip"},
    },
    "musique": {  # the official zip is on Google Drive; a mirror of it
        "kind": "hf", "repo": "bdsaglam/musique",
        "revision": "22873a405dd809893b22ada0b499299fb612d2df",
        "files": ["musique_full_v1.0_train.jsonl",
                  "musique_full_v1.0_dev.jsonl", "README.md"],
    },
    "quality": {
        "kind": "url", "revision": "nyu-mll/quality@f84977c4 v1.0.1",
        "files": {"QuALITY.v1.0.1.zip":
                  "https://github.com/nyu-mll/quality/raw/"
                  "f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1/"
                  "QuALITY.v1.0.1.zip"},
    },
    "nimble": {  # Bespoke Labs' Nimble: train.jsonl and eval.jsonl
        "kind": "url", "revision": "bespokelabsai/nimble@4edb0703",
        "files": {
            f"{s}.jsonl": "https://raw.githubusercontent.com/bespokelabsai/"
            "nimble/4edb0703a2e75d0170c3a45645a1961f4d1a2867/data/"
            f"{s}.jsonl"
            for s in ("train", "eval")
        },
    },
    "contractnli": {  # read from the zip: contract-nli/{train,dev}.json
        "kind": "url", "revision": "Stanford contract-nli.zip (unversioned)",
        "files": {"contract-nli.zip":
                  "https://stanfordnlp.github.io/contract-nli/resources/"
                  "contract-nli.zip"},
    },
}

BENCH_SOURCES: dict[str, dict[str, Any]] = {
    "boolq": {"kind": "hf", "repo": "google/boolq",
              "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              "files": ["data/validation-00000-of-00001.parquet"]},
    "mmlu": {"kind": "hf", "repo": "cais/mmlu",
             "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
             "files": ["all/test-00000-of-00001.parquet",
                       "all/validation-00000-of-00001.parquet",
                       "all/dev-00000-of-00001.parquet"]},
    "mmlu_pro": {"kind": "hf", "repo": "TIGER-Lab/MMLU-Pro",
                 "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                 "files": ["data/test-00000-of-00001.parquet",
                           "data/validation-00000-of-00001.parquet"]},
    "arc_challenge": {"kind": "hf", "repo": "allenai/ai2_arc",
                      "revision": "210d026faf9955653af8916fad021475a3f00453",
                      "files": ["ARC-Challenge/test-00000-of-00001.parquet"]},
    "winogrande": {"kind": "hf", "repo": "allenai/winogrande",
                   "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                   "files": ["winogrande_xl/validation-00000-of-00001.parquet"]},
    "sciq": {"kind": "hf", "repo": "allenai/sciq",
             "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
             "files": ["data/test-00000-of-00001.parquet"]},
    "banking77": {
        "kind": "url", "revision": "PolyAI-LDN/task-specific-datasets@57ec275d",
        "files": {"test.csv":
                  "https://raw.githubusercontent.com/PolyAI-LDN/"
                  "task-specific-datasets/"
                  "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/"
                  "test.csv"}},
    "jevbench": {
        "kind": "url", "revision": "fstandhartinger/jevbench v1.4.2.2 "
        "(commit fcc69279)",
        "files": {
            f"{s}.jsonl": "https://raw.githubusercontent.com/fstandhartinger/"
            "jevbench/fcc69279a933f882ccc4489eb14bf467647dbdb7/datasets/"
            f"public/{s}.jsonl"
            for s in ("easy", "original", "hard")
        }},
    "gsm8k": {"kind": "hf", "repo": "openai/gsm8k",
              "revision": "740312add88f781978c0658806c59bc2815b9866",
              "files": ["main/test-00000-of-00001.parquet"]},
    "ifeval": {"kind": "hf", "repo": "google/IFEval",
               "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
               "files": ["ifeval_input_data.jsonl"]},
    "triviaqa": {"kind": "hf", "repo": "mandarjoshi/trivia_qa",
                 "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
                 "files": ["rc.nocontext/validation-00000-of-00001.parquet"]},
    "lambada": {"kind": "hf", "repo": "EleutherAI/lambada_openai",
                "revision": "900124bf3b8235c6daf21033af9948b3f07346c4",
                "files": ["default/test/default.parquet"]},
    "wikitext2": {"kind": "hf", "repo": "Salesforce/wikitext",
                  "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                  "files": ["wikitext-2-raw-v1/test-00000-of-00001.parquet"]},
    "mmbench": {"kind": "hf", "repo": "lmms-lab/MMBench",
                "revision": "56ba1af8954932c4804bd3f522e05ed96e63b654",
                "files": ["en/dev-00000-of-00001.parquet"]},
    "mmstar": {"kind": "hf", "repo": "Lin-Chen/MMStar",
               "revision": "bc98d668301da7b14f648724866e57302778ab27",
               "files": ["mmstar.parquet"]},
    "dolly": {"kind": "hf", "repo": "databricks/databricks-dolly-15k",
              "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
              "files": ["databricks-dolly-15k.jsonl"]},
    "mt_bench": {"kind": "hf", "repo": "HuggingFaceH4/mt_bench_prompts",
                 "revision": "e3a795c5e9a82ee40611c416b8a7786c73198991",
                 "files": ["raw/question.jsonl"]},
}
# sha256 of every "url" file, checked when it is downloaded. Several of the
# releases are unversioned, so this pins them to the content used so far.
CHECKSUMS: dict[str, dict[str, str]] = {
    "anli": {
        "alphanli-train-dev.zip":
        "24840b27553e93ec625ae020dbf78d92daeae4be31ebbd469a0c9f6f99ed1c8d",
    },
    "contractnli": {
        "contract-nli.zip":
        "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b",
    },
    "cosmosqa": {
        "train.csv":
        "d8d5ca1f9f6534b6530550718591af89372d976a8fc419360fab4158dee4d0b2",
        "valid.csv":
        "a6a94fc1463ca82bb10f98ef68ed535405e6f5c36e044ff8e136b5c19dea63f3",
    },
    "logiqa2": {
        "train.txt":
        "98eb412e8ed53b3d65da5ef75b00b7a0bbdea7970c05ad699291a2a0510922de",
        "dev.txt":
        "bbefb563b7ddc02640ccdc314c1315d5727dba48539d0ecdd126fa351e511b09",
    },
    "massive": {
        "amazon-massive-dataset-1.1.tar.gz":
        "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
    },
    "nimble": {
        "train.jsonl":
        "beadbb9b81837f7c339e090cd210ce91f65a55f2a3ada1ce9b623765b8d6fe2e",
        "eval.jsonl":
        "8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896",
    },
    "piqa": {
        "train.jsonl":
        "f805f9c96518d7b59f2f870d90f712457cda949979469d9fa3fd706ce22e0a1a",
        "train-labels.lst":
        "df8ae080c41f02f88e81f6932d425f4b7ee72f433ad552e6a30d45e358e8f8bb",
        "valid.jsonl":
        "93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d",
        "valid-labels.lst":
        "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb",
    },
    "proofwriter": {
        "proofwriter-dataset-V2020.12.3.zip":
        "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26",
    },
    "quality": {
        "QuALITY.v1.0.1.zip":
        "58552970c3dc4fe199654579e9f74139becd90fca61ddc2eb0b0d82cbb437341",
    },
    "benchmarks/banking77": {
        "test.csv":
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    },
    "benchmarks/jevbench": {
        "easy.jsonl":
        "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
        "original.jsonl":
        "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
        "hard.jsonl":
        "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
    },
}

SUITES = {
    "external": ("boolq", "mmlu", "mmlu_pro", "arc_challenge", "winogrande",
                 "sciq", "banking77"),
    "jevbench": ("jevbench",),
    "general": ("gsm8k", "ifeval", "triviaqa", "lambada", "wikitext2"),
    "image": ("mmbench", "mmstar"),
    "behavior": ("dolly", "mt_bench"),
}


def provenance(entry: dict[str, Any]) -> dict[str, Any]:
    """The pinned revision and where a TRAIN_SOURCES / BENCH_SOURCES entry
    comes from (its repo, else its file URLs), for the manifest."""
    return {"revision": entry["revision"],
            "source": entry.get("repo") or list(entry["files"].values())}


def benchmark_path(name: str, filename: str) -> pathlib.Path:
    """The local path of one file of a benchmark (fetched if missing)."""
    return local_path(f"benchmarks/{name}", filename)


def data_home() -> pathlib.Path:
    """The folder of the datasets that are not in the Hugging Face cache."""
    if override := os.environ.get("LLM_JEV_DATA"):
        return pathlib.Path(override)
    return pathlib.Path(constants.HF_HOME) / "llm_jev"


def _source(name: str) -> dict[str, Any] | None:
    """The TRAIN_SOURCES / BENCH_SOURCES entry of a dataset, if it has one."""
    if name.startswith("benchmarks/"):
        return BENCH_SOURCES.get(name.removeprefix("benchmarks/"))
    return TRAIN_SOURCES.get(name)


def root(name: str) -> pathlib.Path:
    """The folder holding a dataset's raw files, fetching it if needed.

    Args:
      name: A dataset name; benchmarks are "benchmarks/<name>".

    Returns:
      For a "hf" dataset the snapshot folder in the Hugging Face cache (its
      files are fetched on first use), otherwise <data home>/<name>.
    """
    source = _source(name)
    if source is None or source["kind"] != "hf":
        return data_home() / name
    return _snapshot(name)


@functools.cache
def _snapshot(name: str) -> pathlib.Path:
    """The snapshot folder of a "hf" dataset, resolved once per process."""
    from huggingface_hub import snapshot_download  # pylint: disable=C0415

    source = _source(name)
    kwargs = dict(repo_id=source["repo"], repo_type="dataset",
                  revision=source["revision"],
                  allow_patterns=source["files"])
    try:
        return pathlib.Path(snapshot_download(local_files_only=True, **kwargs))
    except Exception:  # pylint: disable=broad-except
        # Not in the cache yet (a missing file raises a Hub error).
        return pathlib.Path(snapshot_download(**kwargs))


def local_path(name: str, filename: str) -> pathlib.Path:
    """The local path of one raw file of a dataset, fetching it if missing.

    Args:
      name: A dataset name; benchmarks are "benchmarks/<name>".
      filename: One of the dataset's files.

    Returns:
      The path in the Hugging Face cache (kind "hf") or in the data home
      (kind "url").

    Raises:
      FileNotFoundError: If a "url" file is missing and cannot be downloaded.
    """
    source = _source(name)
    if source is not None and source["kind"] == "url":
        return fetch(name, filename)
    return root(name) / filename


def sha256(path: pathlib.Path) -> str:
    """The sha256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(name: str, filename: str, path: pathlib.Path) -> None:
    """Checks a "url" file against its pinned sha256.

    Args:
      name: The dataset name; benchmarks are "benchmarks/<name>".
      filename: The file's name in the dataset.
      path: The local file.

    Raises:
      ValueError: If the file's sha256 differs from the pinned one (the
        upstream file changed).
    """
    expected = CHECKSUMS.get(name, {}).get(filename)
    if expected is None:
        return
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path}: sha256 {actual} is not the pinned {expected}; the "
            "upstream file changed. Check it, then update CHECKSUMS.")


def _download_url(url: str, path: pathlib.Path) -> None:
    """Downloads url to path through a .part file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, \
            open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    tmp.rename(path)


def fetch(
    name: str, filename: str, directory: pathlib.Path | None = None
) -> pathlib.Path:
    """The local path of one pinned "url" file, downloading it if missing.

    Args:
      name: A "url" dataset; benchmarks are "benchmarks/<name>".
      filename: One of that source's files.
      directory: The dataset folder; defaults to <data home>/<name>.

    Returns:
      The local path of the file.

    Raises:
      FileNotFoundError: If the file is missing and cannot be downloaded
        (the message names the path and the URL).
    """
    source = _source(name)
    if source is None or source["kind"] != "url":
        raise KeyError(f"{name!r} is not a pinned \"url\" dataset")
    directory = directory or data_home() / name
    path = directory / filename
    if path.exists():
        return path
    url = source["files"][filename]
    try:
        _download_url(url, path)
    except (urllib.error.URLError, OSError) as e:
        raise FileNotFoundError(
            f"{path} is missing and could not be downloaded from {url}: {e}"
        ) from e
    try:
        verify(name, filename, path)
    except ValueError:
        path.unlink()
        raise
    logger.info("downloaded {} from {}", path, url)
    _record(directory, source, {filename: url})
    return path


def _record(
    directory: pathlib.Path, source: dict[str, Any], urls: dict[str, str]
) -> None:
    """Adds files (name -> url) to <directory>/SOURCES.json with their sha256."""
    record_path = directory / "SOURCES.json"
    record = (json.loads(record_path.read_text()) if record_path.exists()
              else {"revision": source["revision"], "files": {}})
    for filename, url in urls.items():
        record["files"][filename] = {
            "url": url, "sha256": sha256(directory / filename)}
    record_path.write_text(json.dumps(record, indent=1) + "\n")


def download(names: Iterable[str] | None = None) -> None:
    """Fetches the pinned training and benchmark files (skips existing).

    Args:
      names: Datasets to fetch; defaults to all.
    """
    sources = {**TRAIN_SOURCES,
               **{f"benchmarks/{k}": v for k, v in BENCH_SOURCES.items()}}
    for name, source in sources.items():
        if names and name.split("/")[-1] not in names:
            continue
        directory = data_home() / name
        directory.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "revision": source["revision"], "files": {}}
        if source["kind"] == "hf":
            record["repo"] = source["repo"]
            snapshot = root(name)
            for f in source["files"]:
                record["files"][f] = sha256(snapshot / f)
        else:
            for f, url in source["files"].items():
                if not (directory / f).exists():
                    _download_url(url, directory / f)
                verify(name, f, directory / f)
                record["files"][f] = {
                    "url": url, "sha256": sha256(directory / f)}
        (directory / "SOURCES.json").write_text(
            json.dumps(record, indent=1) + "\n")
        logger.info("{}: {} files", name, len(record["files"]))
