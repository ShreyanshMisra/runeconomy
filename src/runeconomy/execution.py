"""Shared load-aware execution for static and agent-process experiments.

The pipeline is ``RunSpec -> ArrivalSchedule -> LoadRunner -> TrialExecutor ->
ArtifactStore``; decisions are built afterwards from the recorded trials by
the mode's summary code plus :mod:`runeconomy.metric_gates`.

``docs/load-pairing-and-tail-gates.md`` is the normative description of the
arrival patterns, pairing designs, and frozen schedule implemented here.
Sequential experiments do not use this module; they keep their original
execution loops so recorded contracts reproduce byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence


SEQUENTIAL = "sequential"
CLOSED_LOOP = "closed_loop"
CONSTANT_RPS = "constant_rps"
POISSON = "poisson"
OPEN_LOOP_PATTERNS = frozenset({CONSTANT_RPS, POISSON})
CONCURRENT_ARRIVAL_PATTERNS = frozenset({CLOSED_LOOP, *OPEN_LOOP_PATTERNS})
ARRIVAL_PATTERNS = frozenset({SEQUENTIAL, *CONCURRENT_ARRIVAL_PATTERNS})

VARIANT_BLOCKED = "variant_blocked"
INTERLEAVED = "interleaved"
PAIRING_DESIGNS = frozenset({VARIANT_BLOCKED, INTERLEAVED})
SHARED = "shared"
ISOLATED = "isolated"
SERVING_TOPOLOGIES = frozenset({SHARED, ISOLATED})
DRAIN = "drain"
SUSTAIN = "sustain"
TAIL_MODES = frozenset({DRAIN, SUSTAIN})

ROLE_POPULATION = "population"
ROLE_WARMUP = "warmup"
ROLE_FILLER = "load_filler"

SCHEDULE_VERSION = "0.1"
DEFAULT_MAX_IN_FLIGHT = 256
DEFAULT_MAX_DISPATCH_LAG_MS = 250.0
VARIANTS = ("baseline", "candidate")


class LoadSpecError(ValueError):
    """A load declaration or arrival schedule is invalid."""


def _sub_seed(seed: int, *parts: Any) -> int:
    material = ":".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class LoadSpec:
    """Validated parameters of a concurrent arrival pattern."""

    arrival_pattern: str
    levels: tuple[float, ...]
    gate_level: float
    pairing: str
    serving_topology: str
    block_units: int | None = None
    tail: str = DRAIN
    max_in_flight: int | None = None
    max_dispatch_lag_ms: float | None = None
    schedule_sha256: str | None = None

    @property
    def open_loop(self) -> bool:
        return self.arrival_pattern in OPEN_LOOP_PATTERNS

    @property
    def level_name(self) -> str:
        return "rate_per_second" if self.open_loop else "concurrency"

    @property
    def gate_stage(self) -> int:
        return self.levels.index(self.gate_level)

    @property
    def is_sweep(self) -> bool:
        return len(self.levels) > 1

    @classmethod
    def from_value(
        cls,
        arrival_pattern: Any,
        value: Any,
        *,
        location: str = "execution.load",
        forced_topology: str | None = None,
    ) -> "LoadSpec":
        if arrival_pattern not in CONCURRENT_ARRIVAL_PATTERNS:
            raise LoadSpecError(
                "execution.arrival_pattern must be sequential, closed_loop, "
                "constant_rps, or poisson"
            )
        if not isinstance(value, Mapping):
            raise LoadSpecError(
                f"{location} must be an object for arrival_pattern {arrival_pattern!r}"
            )
        allowed = {
            "pairing",
            "serving_topology",
            "block_units",
            "tail",
            "schedule_sha256",
        }
        if arrival_pattern == CLOSED_LOOP:
            single, multiple, gate = "concurrency", "concurrency_levels", "gate_concurrency"
        else:
            single, multiple, gate = "rate_per_second", "rate_levels", "gate_rate"
            allowed |= {"max_in_flight", "max_dispatch_lag_ms"}
        allowed |= {single, multiple, gate}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise LoadSpecError(
                f"{location} has unsupported fields for {arrival_pattern}: "
                + ", ".join(unknown)
            )

        def level(raw: Any, name: str) -> float:
            if arrival_pattern == CLOSED_LOOP:
                if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
                    raise LoadSpecError(f"{location}.{name} must be an integer >= 1")
                return raw
            if not _number(raw) or not math.isfinite(float(raw)) or float(raw) <= 0:
                raise LoadSpecError(f"{location}.{name} must be a positive number")
            return float(raw)

        if (single in value) == (multiple in value):
            raise LoadSpecError(f"{location} must declare exactly one of {single} or {multiple}")
        if single in value:
            if gate in value:
                raise LoadSpecError(f"{location}.{gate} is only used with {multiple}")
            levels = (level(value[single], single),)
            gate_level = levels[0]
        else:
            raw_levels = value[multiple]
            if not isinstance(raw_levels, list) or not raw_levels:
                raise LoadSpecError(f"{location}.{multiple} must be a non-empty array")
            parsed = [level(item, multiple) for item in raw_levels]
            if len(set(parsed)) != len(parsed):
                raise LoadSpecError(f"{location}.{multiple} must not repeat a level")
            levels = tuple(sorted(parsed))
            if gate not in value:
                raise LoadSpecError(
                    f"{location}.{gate} must preregister which sweep level is gated"
                )
            gate_level = level(value[gate], gate)
            if gate_level not in levels:
                raise LoadSpecError(f"{location}.{gate} must be one of {multiple}")

        topology = value.get("serving_topology", forced_topology)
        if forced_topology is not None and topology != forced_topology:
            raise LoadSpecError(
                f"{location}.serving_topology must be {forced_topology!r} for this mode"
            )
        if topology not in SERVING_TOPOLOGIES:
            raise LoadSpecError(
                f"{location}.serving_topology must declare shared or isolated serving"
            )
        pairing = value.get("pairing", VARIANT_BLOCKED)
        if pairing not in PAIRING_DESIGNS:
            raise LoadSpecError(f"{location}.pairing must be variant_blocked or interleaved")
        if pairing == INTERLEAVED and topology != ISOLATED:
            raise LoadSpecError(
                f"{location}.pairing interleaved mixes both variants in flight and "
                "requires serving_topology 'isolated'; shared serving must use "
                "variant_blocked"
            )
        block_units = value.get("block_units")
        if block_units is not None:
            if pairing != VARIANT_BLOCKED:
                raise LoadSpecError(f"{location}.block_units applies only to variant_blocked")
            if (
                not isinstance(block_units, int)
                or isinstance(block_units, bool)
                or block_units < 1
            ):
                raise LoadSpecError(f"{location}.block_units must be an integer >= 1")
        tail = value.get("tail", DRAIN)
        if tail not in TAIL_MODES:
            raise LoadSpecError(f"{location}.tail must be drain or sustain")
        max_in_flight: int | None = None
        max_lag: float | None = None
        if arrival_pattern in OPEN_LOOP_PATTERNS:
            max_in_flight = value.get("max_in_flight", DEFAULT_MAX_IN_FLIGHT)
            if (
                not isinstance(max_in_flight, int)
                or isinstance(max_in_flight, bool)
                or max_in_flight < 1
            ):
                raise LoadSpecError(f"{location}.max_in_flight must be an integer >= 1")
            raw_lag = value.get("max_dispatch_lag_ms", DEFAULT_MAX_DISPATCH_LAG_MS)
            if not _number(raw_lag) or float(raw_lag) < 0:
                raise LoadSpecError(f"{location}.max_dispatch_lag_ms must be non-negative")
            max_lag = float(raw_lag)
        pin = value.get("schedule_sha256")
        if pin is not None and (
            not isinstance(pin, str)
            or not pin.startswith("sha256:")
            or len(pin) != 71
        ):
            raise LoadSpecError(f"{location}.schedule_sha256 must be a sha256: digest")
        return cls(
            arrival_pattern=arrival_pattern,
            levels=levels,
            gate_level=gate_level,
            pairing=pairing,
            serving_topology=topology,
            block_units=block_units,
            tail=tail,
            max_in_flight=max_in_flight,
            max_dispatch_lag_ms=max_lag,
            schedule_sha256=pin,
        )

    def to_dict(self, *, include_pin: bool = True) -> dict[str, Any]:
        """Return the canonical manifest form; it parses back to an equal spec."""

        if self.arrival_pattern == CLOSED_LOOP:
            single, multiple, gate = "concurrency", "concurrency_levels", "gate_concurrency"
        else:
            single, multiple, gate = "rate_per_second", "rate_levels", "gate_rate"
        payload: dict[str, Any] = {
            "pairing": self.pairing,
            "serving_topology": self.serving_topology,
            "tail": self.tail,
        }
        if self.is_sweep:
            payload[multiple] = list(self.levels)
            payload[gate] = self.gate_level
        else:
            payload[single] = self.levels[0]
        if self.block_units is not None:
            payload["block_units"] = self.block_units
        if self.open_loop:
            payload["max_in_flight"] = self.max_in_flight
            payload["max_dispatch_lag_ms"] = self.max_dispatch_lag_ms
        if include_pin and self.schedule_sha256 is not None:
            payload["schedule_sha256"] = self.schedule_sha256
        return payload

    def claim_boundary(self) -> str:
        if self.arrival_pattern == CLOSED_LOOP:
            load = f"closed-loop concurrency {int(self.gate_level)} per variant"
        else:
            kind = "Poisson" if self.arrival_pattern == POISSON else "constant-rate"
            load = f"{kind} arrivals at {self.gate_level:g} units/s per variant"
        design = (
            "variant-blocked on shared serving"
            if self.pairing == VARIANT_BLOCKED
            else "interleaved on isolated serving"
        )
        return f"{load}, {design}, tail {self.tail}"


@dataclass(frozen=True)
class ScheduledTrial:
    """One frozen dispatch slot. ``unit`` indexes the run's unit list."""

    index: int
    stage: int
    level: float
    block: int
    variant: str
    unit: int
    repetition: int
    position: int
    offset_ms: float | None
    warmup: bool

    @property
    def role(self) -> str:
        return ROLE_WARMUP if self.warmup else ROLE_POPULATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "level": self.level,
            "block": self.block,
            "variant": self.variant,
            "unit": self.unit,
            "repetition": self.repetition,
            "position": self.position,
            "offset_ms": self.offset_ms,
            "warmup": self.warmup,
        }


@dataclass(frozen=True)
class ScheduleBlock:
    """A barrier-delimited group of trials run under one load level."""

    block: int
    stage: int
    level: float
    block_pair: int | None
    variants: tuple[str, ...]
    warmup: bool
    trials: tuple[ScheduledTrial, ...]
    filler_seed: int

    def pool(self, variant: str) -> list[ScheduledTrial]:
        return [trial for trial in self.trials if trial.variant == variant]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "stage": self.stage,
            "level": self.level,
            "block_pair": self.block_pair,
            "variants": list(self.variants),
            "warmup": self.warmup,
            "trial_indexes": [trial.index for trial in self.trials],
        }


@dataclass(frozen=True)
class ArrivalSchedule:
    load: LoadSpec
    unit_ids: tuple[str, ...]
    repetitions: int
    warmup_repetitions: int
    seed: int
    order: str
    blocks: tuple[ScheduleBlock, ...]

    @property
    def trials(self) -> list[ScheduledTrial]:
        return [trial for block in self.blocks for trial in block.trials]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_version": SCHEDULE_VERSION,
            "load": self.load.to_dict(include_pin=False),
            "arrival_pattern": self.load.arrival_pattern,
            "unit_ids": list(self.unit_ids),
            "repetitions": self.repetitions,
            "warmup_repetitions": self.warmup_repetitions,
            "seed": self.seed,
            "order": self.order,
            "gate_stage": self.load.gate_stage,
            "blocks": [block.to_dict() for block in self.blocks],
            "trials": [trial.to_dict() for trial in self.trials],
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        population = [trial for trial in self.trials if not trial.warmup]
        return {
            "schedule_version": SCHEDULE_VERSION,
            "sha256": self.digest(),
            "arrival_pattern": self.load.arrival_pattern,
            "load": self.load.to_dict(),
            "claim_boundary": self.load.claim_boundary(),
            "stage_levels": list(self.load.levels),
            "gate_stage": self.load.gate_stage,
            "gate_level": self.load.gate_level,
            "block_count": len(self.blocks),
            "block_pair_count": len(
                {
                    (block.stage, block.block_pair)
                    for block in self.blocks
                    if block.block_pair is not None
                }
            ),
            "population_trials": len(population),
            "gate_stage_population_trials": sum(
                trial.stage == self.load.gate_stage for trial in population
            ),
            "warmup_trials": len(self.trials) - len(population),
        }


def _offsets(load: LoadSpec, level: float, count: int, seed: int) -> list[float | None]:
    if not load.open_loop:
        return [None] * count
    interval_ms = 1000.0 / level
    if load.arrival_pattern == CONSTANT_RPS:
        return [round(index * interval_ms, 6) for index in range(count)]
    rng = random.Random(seed)
    current = 0.0
    values: list[float | None] = []
    for index in range(count):
        if index:
            current += rng.expovariate(level) * 1000.0
        values.append(round(current, 6))
    return values


def filler_offsets(load: LoadSpec, block: ScheduleBlock) -> Iterator[float]:
    """Continue a block's open-loop arrival process for sustain-tail fillers."""

    population = block.pool(block.variants[0])
    last = population[-1].offset_ms if population and population[-1].offset_ms is not None else 0.0
    rng = random.Random(block.filler_seed)
    current = float(last)
    interval_ms = 1000.0 / block.level
    while True:
        if load.arrival_pattern == CONSTANT_RPS:
            current += interval_ms
        else:
            current += rng.expovariate(block.level) * 1000.0
        yield round(current, 6)


def build_arrival_schedule(
    load: LoadSpec,
    unit_ids: Sequence[str],
    *,
    repetitions: int,
    seed: int,
    order: str,
    warmup_repetitions: int = 0,
) -> ArrivalSchedule:
    """Build the deterministic frozen schedule for a concurrent experiment."""

    if not unit_ids:
        raise LoadSpecError("a load schedule requires at least one unit")
    if repetitions < 1:
        raise LoadSpecError("a load schedule requires at least one repetition")
    if order not in {"alternating", "randomized"}:
        raise LoadSpecError("execution.order must be alternating or randomized")
    units = len(unit_ids)
    blocks: list[ScheduleBlock] = []
    trials_index = 0

    def add_block(
        *,
        stage: int,
        level: float,
        block_pair: int | None,
        variants: tuple[str, ...],
        slots: list[tuple[int, int]],
        offsets: list[float | None],
        warmup: bool,
    ) -> None:
        nonlocal trials_index
        block_number = len(blocks)
        trials: list[ScheduledTrial] = []
        for variant in variants:
            for position, ((repetition, unit), offset) in enumerate(zip(slots, offsets)):
                trials.append(
                    ScheduledTrial(
                        index=trials_index,
                        stage=stage,
                        level=level,
                        block=block_number,
                        variant=variant,
                        unit=unit,
                        repetition=repetition,
                        position=position,
                        offset_ms=offset,
                        warmup=warmup,
                    )
                )
                trials_index += 1
        blocks.append(
            ScheduleBlock(
                block=block_number,
                stage=stage,
                level=level,
                block_pair=block_pair,
                variants=variants,
                warmup=warmup,
                trials=tuple(trials),
                filler_seed=_sub_seed(seed, "filler", block_number),
            )
        )

    for stage, level in enumerate(load.levels):
        if not load.open_loop and units * repetitions < level:
            raise LoadSpecError(
                f"concurrency {int(level)} exceeds the {units * repetitions} "
                "population units per variant; the declared load can never be reached"
            )
        if warmup_repetitions:
            warm_slots = [
                (repetition, unit)
                for repetition in range(warmup_repetitions)
                for unit in range(units)
            ]
            random.Random(_sub_seed(seed, "warmup-order", stage)).shuffle(warm_slots)
            warm_offsets = _offsets(
                load, level, len(warm_slots), _sub_seed(seed, "warmup-offsets", stage)
            )
            if load.pairing == INTERLEAVED:
                add_block(
                    stage=stage,
                    level=level,
                    block_pair=None,
                    variants=VARIANTS,
                    slots=warm_slots,
                    offsets=warm_offsets,
                    warmup=True,
                )
            else:
                for variant in VARIANTS:
                    add_block(
                        stage=stage,
                        level=level,
                        block_pair=None,
                        variants=(variant,),
                        slots=warm_slots,
                        offsets=warm_offsets,
                        warmup=True,
                    )
        slots = [
            (repetition, unit)
            for repetition in range(repetitions)
            for unit in range(units)
        ]
        if load.pairing == INTERLEAVED:
            random.Random(_sub_seed(seed, "order", stage, 0)).shuffle(slots)
            add_block(
                stage=stage,
                level=level,
                block_pair=0,
                variants=VARIANTS,
                slots=slots,
                offsets=_offsets(load, level, len(slots), _sub_seed(seed, "offsets", stage, 0)),
                warmup=False,
            )
            continue
        size = load.block_units or units
        chunks = [slots[start : start + size] for start in range(0, len(slots), size)]
        if not load.open_loop:
            short = [len(chunk) for chunk in chunks if len(chunk) < level]
            if short:
                raise LoadSpecError(
                    f"a block of {short[0]} units cannot sustain concurrency "
                    f"{int(level)}; choose block_units so every block has at least "
                    "that many units"
                )
        coin = random.Random(_sub_seed(seed, "block-order", stage))
        for pair, chunk in enumerate(chunks):
            dispatch = list(chunk)
            random.Random(_sub_seed(seed, "order", stage, pair)).shuffle(dispatch)
            offsets = _offsets(
                load, level, len(dispatch), _sub_seed(seed, "offsets", stage, pair)
            )
            if order == "alternating":
                arms = VARIANTS if pair % 2 == 0 else tuple(reversed(VARIANTS))
            else:
                arms = VARIANTS if coin.random() < 0.5 else tuple(reversed(VARIANTS))
            for variant in arms:
                add_block(
                    stage=stage,
                    level=level,
                    block_pair=pair,
                    variants=(variant,),
                    slots=dispatch,
                    offsets=offsets,
                    warmup=False,
                )
    return ArrivalSchedule(
        load=load,
        unit_ids=tuple(unit_ids),
        repetitions=repetitions,
        warmup_repetitions=warmup_repetitions,
        seed=seed,
        order=order,
        blocks=tuple(blocks),
    )


def verify_schedule_pin(schedule: ArrivalSchedule) -> None:
    pin = schedule.load.schedule_sha256
    if pin is not None and pin != schedule.digest():
        raise LoadSpecError(
            "execution.load.schedule_sha256 does not match the arrival schedule "
            f"built from this manifest (built {schedule.digest()})"
        )


class TrialExecutor(Protocol):
    """Executes one scheduled trial and returns its record.

    ``filler_index`` is set only for ``load_filler`` trials, which reuse a
    population slot's unit but must not collide with its artifacts.
    Implementations are called concurrently from worker threads.
    """

    def execute(
        self, trial: ScheduledTrial, *, role: str, filler_index: int | None
    ) -> dict[str, Any]: ...


class ArtifactStore(Protocol):
    def append(self, record: Mapping[str, Any]) -> None: ...


class LoadObserver(Protocol):
    """Optional hooks for server telemetry. Every method is optional.

    ``on_block_start`` / ``on_block_end`` run on the scheduler thread at block
    barriers; a mapping returned by ``on_block_end`` is attached to the block
    record as ``telemetry``. ``on_trial_end`` runs on worker threads and must
    be thread-safe; a mapping it returns is stored on the trial record as
    ``server_telemetry``.
    """


class JsonlArtifactStore:
    """Append-only, fsynced, thread-safe JSONL log of completed records."""

    def __init__(self, path: Path, header: Mapping[str, Any] | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if header is not None:
            self._write(header, "x")

    def _write(self, record: Mapping[str, Any], mode: str) -> None:
        with self.path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._write(record, "a")


@dataclass
class LoadRunResult:
    records: list[dict[str, Any]]
    fillers: list[dict[str, Any]]
    blocks: list[dict[str, Any]] = field(default_factory=list)


def _call_observer(observer: Any, name: str, *args: Any) -> Any:
    method = getattr(observer, name, None)
    return method(*args) if callable(method) else None


def _time_weighted_in_flight(
    target: tuple[float, float], intervals: list[tuple[float, float]]
) -> float | None:
    start, end = target
    span = end - start
    if span <= 0:
        return None
    covered = 0.0
    for other_start, other_end in intervals:
        overlap = min(end, other_end) - max(start, other_start)
        if overlap > 0:
            covered += overlap
    return covered / span


class LoadRunner:
    """Run an :class:`ArrivalSchedule` with one worker pool per variant."""

    def __init__(
        self,
        schedule: ArrivalSchedule,
        executor: TrialExecutor,
        *,
        store: ArtifactStore | None = None,
        observers: Sequence[Any] = (),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.schedule = schedule
        self.executor = executor
        self.store = store
        self.observers = tuple(observers)
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._errors: list[BaseException] = []
        self._filler_counter = 0
        self._origin = 0.0

    def _elapsed_ms(self, instant: float) -> float:
        return round((instant - self._origin) * 1000.0, 3)

    def _next_filler_index(self) -> int:
        with self._lock:
            value = self._filler_counter
            self._filler_counter += 1
            return value

    def _execute(
        self,
        block: ScheduleBlock,
        trial: ScheduledTrial,
        role: str,
        in_flight: dict[str, int],
        scheduled_at: float | None,
        filler_index: int | None,
    ) -> dict[str, Any] | None:
        with self._lock:
            in_flight[trial.variant] += 1
            at_start = in_flight[trial.variant]
        started = self.clock()
        try:
            record = self.executor.execute(trial, role=role, filler_index=filler_index)
            ended = self.clock()
            telemetry: dict[str, Any] = {}
            for observer in self.observers:
                value = _call_observer(observer, "on_trial_end", trial, record)
                if isinstance(value, Mapping):
                    telemetry.update(value)
            if telemetry:
                record["server_telemetry"] = telemetry
        except BaseException as exc:  # Surface after the barrier.
            with self._lock:
                self._errors.append(exc)
            self._stop.set()
            return None
        finally:
            with self._lock:
                in_flight[trial.variant] -= 1
        lag = None
        if scheduled_at is not None:
            lag = round(max(0.0, started - scheduled_at) * 1000.0, 3)
        record["load"] = {
            "role": role,
            "stage": trial.stage,
            "level": trial.level,
            "block": trial.block,
            "position": trial.position,
            "schedule_index": trial.index,
            "filler_index": filler_index,
            "scheduled_offset_ms": trial.offset_ms,
            "dispatch_lag_ms": lag,
            "started_ms": self._elapsed_ms(started),
            "ended_ms": self._elapsed_ms(ended),
            "in_flight_at_start": at_start,
        }
        if self.store is not None:
            self.store.append(
                {
                    "record_type": "load_filler" if role == ROLE_FILLER else "trial",
                    "record": record,
                }
            )
        return record

    def _run_closed_block(self, block: ScheduleBlock) -> list[dict[str, Any]]:
        level = int(block.level)
        queues = {variant: deque(block.pool(variant)) for variant in block.variants}
        filler_cursor = {variant: 0 for variant in block.variants}
        remaining = {variant: len(queues[variant]) for variant in block.variants}
        in_flight = {variant: 0 for variant in block.variants}
        completed: list[dict[str, Any]] = []
        sustain = self.schedule.load.tail == SUSTAIN and not block.warmup
        population = {variant: block.pool(variant) for variant in block.variants}

        def worker(variant: str) -> None:
            while not self._stop.is_set():
                with self._lock:
                    if queues[variant]:
                        trial = queues[variant].popleft()
                        role = trial.role
                    elif sustain and remaining[variant] > 0:
                        pool = population[variant]
                        trial = pool[filler_cursor[variant] % len(pool)]
                        filler_cursor[variant] += 1
                        role = ROLE_FILLER
                    else:
                        return
                filler_index = self._next_filler_index() if role == ROLE_FILLER else None
                record = self._execute(block, trial, role, in_flight, None, filler_index)
                if record is None:
                    return
                with self._lock:
                    completed.append(record)
                    if role != ROLE_FILLER:
                        remaining[variant] -= 1

        threads = [
            threading.Thread(target=worker, args=(variant,), daemon=True)
            for variant in block.variants
            for _ in range(level)
        ]
        for thread in threads:
            thread.start()
        self._join(threads)
        return completed

    def _run_open_block(self, block: ScheduleBlock) -> list[dict[str, Any]]:
        load = self.schedule.load
        assert load.max_in_flight is not None
        arrivals = sorted(
            block.trials,
            key=lambda trial: (trial.offset_ms or 0.0, block.variants.index(trial.variant)),
        )
        in_flight = {variant: 0 for variant in block.variants}
        remaining = {variant: len(block.pool(variant)) for variant in block.variants}
        completed: list[dict[str, Any]] = []
        futures: list[Future[Any]] = []
        pools = {
            variant: ThreadPoolExecutor(
                max_workers=load.max_in_flight, thread_name_prefix=f"runeconomy-{variant}"
            )
            for variant in block.variants
        }

        def run(trial: ScheduledTrial, role: str, due: float, filler_index: int | None) -> None:
            if self._stop.is_set():
                return
            if role == ROLE_FILLER:
                with self._lock:
                    # A filler queued behind a saturated pool is pointless once
                    # that pool's population has finished.
                    if remaining[trial.variant] == 0:
                        return
            record = self._execute(block, trial, role, in_flight, due, filler_index)
            if record is None:
                return
            with self._lock:
                completed.append(record)
                if role != ROLE_FILLER:
                    remaining[trial.variant] -= 1

        def wait_until(due: float) -> bool:
            while not self._stop.is_set():
                delay = due - self.clock()
                if delay <= 0:
                    return True
                self.sleep(min(delay, 0.05))
            return False

        start = self.clock()
        try:
            for trial in arrivals:
                due = start + (trial.offset_ms or 0.0) / 1000.0
                if not wait_until(due):
                    break
                futures.append(pools[trial.variant].submit(run, trial, trial.role, due, None))
            if load.tail == SUSTAIN and not block.warmup:
                generator = filler_offsets(load, block)
                cursors = {variant: 0 for variant in block.variants}
                pool_trials = {variant: block.pool(variant) for variant in block.variants}
                while not self._stop.is_set():
                    with self._lock:
                        active = [variant for variant in block.variants if remaining[variant] > 0]
                    if not active:
                        break
                    due = start + next(generator) / 1000.0
                    if not wait_until(due):
                        break
                    for variant in active:
                        trial = pool_trials[variant][cursors[variant] % len(pool_trials[variant])]
                        cursors[variant] += 1
                        futures.append(
                            pools[variant].submit(
                                run, trial, ROLE_FILLER, due, self._next_filler_index()
                            )
                        )
        except BaseException:
            self._stop.set()
            raise
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True)
        return completed

    def _join(self, threads: list[threading.Thread]) -> None:
        try:
            for thread in threads:
                while thread.is_alive():
                    thread.join(0.1)
        except BaseException:
            self._stop.set()
            for thread in threads:
                thread.join()
            raise

    def run(self) -> LoadRunResult:
        self._origin = self.clock()
        records: list[dict[str, Any]] = []
        fillers: list[dict[str, Any]] = []
        block_records: list[dict[str, Any]] = []
        for block in self.schedule.blocks:
            info = block.to_dict()
            for observer in self.observers:
                _call_observer(observer, "on_block_start", info)
            block_started = self.clock()
            completed = (
                self._run_open_block(block)
                if self.schedule.load.open_loop
                else self._run_closed_block(block)
            )
            block_ended = self.clock()
            if self._errors:
                raise self._errors[0]
            for variant in block.variants:
                pool = [record for record in completed if _variant_of(record) == variant]
                intervals = [
                    (record["load"]["started_ms"], record["load"]["ended_ms"])
                    for record in pool
                ]
                for record in pool:
                    mean = _time_weighted_in_flight(
                        (record["load"]["started_ms"], record["load"]["ended_ms"]),
                        intervals,
                    )
                    record["load"]["mean_in_flight"] = (
                        None if mean is None else round(mean, 4)
                    )
            telemetry: dict[str, Any] = {}
            for observer in self.observers:
                value = _call_observer(observer, "on_block_end", info)
                if isinstance(value, Mapping):
                    telemetry.update(value)
            block_record = {
                **info,
                "started_ms": self._elapsed_ms(block_started),
                "ended_ms": self._elapsed_ms(block_ended),
                "population_completed": sum(
                    record["load"]["role"] != ROLE_FILLER for record in completed
                ),
                "filler_completed": sum(
                    record["load"]["role"] == ROLE_FILLER for record in completed
                ),
            }
            if telemetry:
                block_record["telemetry"] = telemetry
            block_records.append(block_record)
            for record in completed:
                (fillers if record["load"]["role"] == ROLE_FILLER else records).append(record)
        records.sort(key=lambda record: record["load"]["schedule_index"])
        fillers.sort(key=lambda record: record["load"]["filler_index"])
        return LoadRunResult(records=records, fillers=fillers, blocks=block_records)


def _variant_of(record: Mapping[str, Any]) -> str | None:
    variant = record.get("variant")
    return variant if isinstance(variant, str) else None
