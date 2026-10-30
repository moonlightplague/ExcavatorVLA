"""Measured repeat-task metrics, using the online evaluator's ID accounting."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import numpy as np


def evaluation_helpers():
    # Isaac already imports a different excavator_common package. Load the pure
    # evaluation helpers under a private package name without changing sys.path.
    name = "baseline._evaluation_common"
    if name not in sys.modules:
        directory = Path(__file__).resolve().parents[1] / "ExcavatorVLA-smolvla/excavator_common"
        spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                    submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from baseline._evaluation_common.evaluation_metrics import ParticleTransferTracker, numeric_summary
    return ParticleTransferTracker, numeric_summary


ParticleTransferTracker, numeric_summary = evaluation_helpers()
SCHEMA = "repeat_episode_metrics_v1"
TRANSFER_FIELDS = ("payload_particles", "scooped_particles", "retained_particles",
                   "spillage_particles", "spillage_fraction")
DEFINITIONS = {
    "particle_accounting": "Online evaluation ParticleTransferTracker; strict initial-pile IDs, no region-count fallback.",
    "payload_particles": "Source IDs currently in truck, excluding IDs in truck at episode start; bucket overlap excluded.",
    "scooped_particles": "Unique source IDs observed in bucket; repeat scoops of an ID count once per episode.",
    "retained_particles": "Scooped source IDs still in bucket at measurement time.",
    "spillage_particles": "Scooped IDs outside bucket and truck, including returned-to-pile and airborne material.",
    "spillage_fraction": "Spillage / unique scooped IDs; null before any scoop.",
    "geometry": "Collector sand_region_masks: physical bucket load volume and current truck polygon/Z range; differs from evaluation truck-local bounding box.",
    "sampling": "At episode start, each successfully recorded source sample (10 Hz), cycle boundaries, and episode finish; transient transfers may be missed.",
    "cycle_time_seconds": "Full repeated episode simulation duration, including deposits and final physical return; null on failure.",
    "cycle_time_wall_seconds": "Full repeated episode wall duration; null on failure.",
    "runtime_seconds": "Wall time from recording start to final measurement, including planning between scoops; excludes initial preparation/planning.",
    "attempt_runtime_seconds": "Wall time including scene preparation and initial planning, excluding simulator startup.",
    "scoop_cycles": "Per attempted scoop: simulation/wall execution time through settling and return; excludes planning before executor entry. Success means execution and return completed; quality gates are reported separately. Cycle payload excludes previously deposited IDs.",
    "mass": "Particle counts times sand_particle_mass in kg; null if unavailable.",
}


class RepeatEpisodeMetrics:
    def __init__(self, rt, attempt_started):
        self.rt = rt
        self.attempt_started = attempt_started
        self.started_wall = time.perf_counter()
        self.started_sim = float(rt.dataset_simulation_time_seconds())
        self.source_mask = np.asarray(rt.STATE["dataset_initial_pile_particle_mask"], dtype=bool).copy()
        self.tracker = ParticleTransferTracker()
        self.transfer = dict.fromkeys(TRANSFER_FIELDS)
        self.error = None
        self.observations = 0
        self.cycle = None
        self.cycles = []
        self.observe()

    def observe(self):
        if self.error is not None:
            return
        try:
            points = self.rt.sand_particle_positions()
            if points is None or len(points) == 0:
                raise ValueError("missing_particle_positions")
            if self.source_mask.shape != (len(points),):
                raise ValueError("source_particle_count_changed")
            if not np.isfinite(points).all():
                raise ValueError("nonfinite_particle_positions")
            _, bucket, truck = self.rt.sand_region_masks(points)
            bucket = np.asarray(bucket, dtype=bool) & self.source_mask
            truck = np.asarray(truck, dtype=bool) & self.source_mask & ~bucket
            bucket_ids = np.flatnonzero(bucket).tolist()
            truck_ids = np.flatnonzero(truck).tolist()
            self.transfer = self.tracker.observe(len(points), bucket_ids, truck_ids)
            if self.cycle is not None:
                self.cycle["transfer"] = self.cycle["tracker"].observe(len(points), bucket_ids, truck_ids)
            self.observations += 1
        except (TypeError, ValueError) as exc:
            # Preserve the collected trajectory, but never publish misleading
            # zeros or stale values if source identity becomes unavailable.
            self.error = str(exc)
            self.transfer = dict.fromkeys(TRANSFER_FIELDS)

    def start_cycle(self, index):
        self.cycle = {"scoop_index": int(index), "started_wall": time.perf_counter(),
                      "started_sim": float(self.rt.dataset_simulation_time_seconds()),
                      "tracker": ParticleTransferTracker(), "transfer": {}}
        self.observe()

    def finish_cycle(self, success, reason):
        self.observe()
        cycle = self.cycle
        row = {"scoop_index": cycle["scoop_index"], "success": bool(success), "failure_reason": str(reason),
               "runtime_seconds": max(0., time.perf_counter() - cycle["started_wall"]),
               "simulated_runtime_seconds": max(0., float(self.rt.dataset_simulation_time_seconds()) - cycle["started_sim"])}
        row.update({field: cycle["transfer"].get(field) if self.error is None else None for field in TRANSFER_FIELDS})
        row["cycle_time_seconds"] = row["simulated_runtime_seconds"] if success else None
        row["cycle_time_wall_seconds"] = row["runtime_seconds"] if success else None
        self.cycles.append(row)
        self.cycle = None
        return row

    def finalize(self, meta, bed):
        self.observe()
        wall = time.perf_counter()
        runtime = max(0., wall - self.started_wall)
        simulated = max(0., float(self.rt.dataset_simulation_time_seconds()) - self.started_sim)
        success = bool(meta.get("success"))
        result = {"metrics_schema_version": SCHEMA, "available": self.error is None,
                  "unavailable_reason": self.error, "definitions": DEFINITIONS,
                  "success": success, "status": meta.get("status"),
                  "failure_reason": meta.get("failure_reason", ""),
                  "runtime_seconds": runtime, "attempt_runtime_seconds": max(0., wall - self.attempt_started),
                  "simulated_runtime_seconds": simulated,
                  "cycle_time_seconds": simulated if success else None,
                  "cycle_time_wall_seconds": runtime if success else None,
                  "source_particle_count": int(np.count_nonzero(self.source_mask)),
                  "particle_observations": self.observations,
                  "samples": int(self.rt.STATE.get("dataset_samples", 0)) - int(self.rt.STATE.get("dataset_episode_sample_start", 0)),
                  "scoops_completed": int(meta.get("scoops_completed", 0)),
                  "returns_completed": int(meta.get("repeat_returns_completed", 0)),
                  "score_summary": meta.get("score_summary", {}),
                  "quality_metrics": meta.get("final_metrics", {}),
                  "scoop_cycles": self.cycles,
                  "scoop_cycle_time_seconds": numeric_summary([row["cycle_time_seconds"] for row in self.cycles]),
                  "bed_fill": bed}
        result.update({field: self.transfer.get(field) for field in TRANSFER_FIELDS})
        result["particle_tracking_method"] = self.transfer.get("method")
        mass = self.rt.sand_particle_mass()
        mass = float(mass) if mass is not None and np.isfinite(mass) and mass > 0 else None
        result["particle_mass_kg"] = mass
        for field in ("payload", "scooped", "retained", "spillage"):
            count = result[field + "_particles"]
            result[field + "_mass_kg"] = count * mass if mass is not None and count is not None else None
        result["payload_particles_per_sim_second"] = result["payload_particles"] / simulated if simulated > 0 and result["payload_particles"] is not None else None
        return result
