"""Parent-process progress bars; resume skips do not inflate generation speed."""

from __future__ import annotations

from time import monotonic

from tqdm import tqdm


def count_inputs(paths, limit: int | None) -> tuple[int, int]:
    """One bounded-memory input pass supplies accurate file/sample totals for ETA."""
    files = samples = 0
    limit = int(limit) if limit is not None else None
    with tqdm(desc="Counting input", unit="file", dynamic_ncols=True, leave=False) as bar:
        for path in paths:
            count = 0
            with path.open("rb") as source:
                for line in source:
                    if line.strip():
                        count += 1
                        if limit is not None and count >= limit:
                            break
            files += 1
            samples += count
            bar.set_postfix(samples=samples, refresh=False)
            bar.update(1)
    return files, samples


class RunProgress:
    """Two shared bars for any number of file processes, rendered only by the parent."""

    def __init__(self, files: int | None, samples: int | None, *, enabled: bool = True):
        self.enabled = enabled
        self.sample_total = samples
        self.stats = {"saved": 0, "resumed": 0, "failed": 0}
        self.active = {}
        self.started = self.last_refresh = monotonic()
        self.samples = tqdm(total=samples, desc="Samples", unit="sample", position=0,
                            dynamic_ncols=True, disable=not enabled,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]")
        self.files = tqdm(total=files, desc="Files", unit="file", position=1,
                          dynamic_ncols=True, disable=not enabled,
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

    def register(self, relative: str) -> None:
        self.active[relative] = {key: 0 for key in self.stats}

    def report(self, relative: str, stats: dict, error: str | None = None) -> None:
        if error:
            tqdm.write(error)
        current = self.active.get(relative)
        if current is None:
            return  # a final job result may arrive before its last queued UI event
        advance = 0
        for key in self.stats:
            delta = max(0, stats.get(key, 0) - current[key])
            current[key] += delta
            self.stats[key] += delta
            advance += delta
        self.samples.update(advance)

    def finish(self, relative: str, stats: dict | None) -> None:
        if stats is not None:
            self.report(relative, stats)
        self.active.pop(relative, None)
        self.files.update(1)

    def estimates(self) -> tuple[float | None, float | None]:
        attempted = self.stats["saved"] + self.stats["failed"]
        rate = attempted / max(monotonic() - self.started, 1e-9) if attempted else None
        remaining = max(0, self.sample_total - sum(self.stats.values())) if self.sample_total is not None else None
        eta = 0.0 if remaining == 0 else remaining / rate if remaining is not None and rate else None
        return rate, eta

    def refresh(self, *, force: bool = False) -> None:
        now = monotonic()
        if not self.enabled or (not force and now - self.last_refresh < 0.5):
            return
        rate, eta = self.estimates()
        self.samples.set_postfix(**self.stats, rate=f"{rate:.3g} sample/s" if rate is not None else "?",
                                 ETA=tqdm.format_interval(eta) if eta is not None else "?", refresh=False)
        self.samples.refresh()
        self.files.refresh()
        self.last_refresh = now

    def __enter__(self):
        self.refresh(force=True)
        return self

    def __exit__(self, *args):
        self.refresh(force=True)
        self.files.close()
        self.samples.close()
