from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from SAEUnitAnalysis.analyses import (
    _categorical_dense_scores,
    _centroid_holdout_metrics,
    _controlled_pair_indices,
    _embed_umap,
    _holdout_metrics_and_confusion,
    _load_umap,
    _paired_cosines,
    clustering_analysis,
    disentanglement_tables,
    geometry_analysis,
    health_analysis,
    phone_unit_confusion,
    phone_speaker_unit_scores,
    selectivity_analysis,
)
from SAEUnitAnalysis.causal import _bootstrap_mode_summary
from SAEUnitAnalysis.bundle import AnalysisBundle
from SAEUnitAnalysis.build_librispeech_bundle import build_bundle as build_librispeech_bundle
from SAEUnitAnalysis.build_timit_bundle import build_bundle
from SAEUnitAnalysis.checkpoint import load_checkpoint, route_information, unresolved_critical
from SAEUnitAnalysis.extraction import (
    FeatureCache, _block_spec, _encode_sparse, _quick_sample,
    _speaker_balanced_sample, parse_split_limits,
)
from SAEUnitAnalysis.factor_metrics import speech_factor_metrics
from SAEUnitAnalysis.import_mfa_alignments import import_alignments, parse_textgrid
from SAEUnitAnalysis.pipeline import run_analysis
from SAEUnitAnalysis.prepare_librispeech_mfa_corpus import prepare_corpus
from SAEUnitAnalysis.types import ResolvedModel
from SAEUnitAnalysis.utils import AnalysisError


def _wav(path: Path, sr: int = 16000) -> None:
    t = np.arange(sr // 5) / sr
    x = (0.15 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())


def _bundle(root: Path, splits=("test",)) -> Path:
    (root / "audio").mkdir(parents=True)
    rows = []
    align = []
    for i, split in enumerate(splits):
        uid = f"u{i}"; _wav(root / "audio" / f"{uid}.wav")
        rows.append({"utterance_id":uid, "audio_path":f"audio/{uid}.wav", "split":split,
                     "transcript":"a test", "speaker_id":f"s{i%2}", "emotion":["neutral","happy"][i%2]})
        align += [{"utterance_id":uid,"start_sec":0.0,"end_sec":0.1,"phone":"AA"},
                  {"utterance_id":uid,"start_sec":0.1,"end_sec":0.2,"phone":"T"}]
    with (root / "utterances.csv").open("w", newline="") as f:
        wri=csv.DictWriter(f, fieldnames=rows[0]); wri.writeheader(); wri.writerows(rows)
    with (root / "alignments.csv").open("w", newline="") as f:
        wri=csv.DictWriter(f, fieldnames=align[0]); wri.writeheader(); wri.writerows(align)
    (root / "dataset.yaml").write_text(json.dumps({
        "schema_version":1,"manifest":"utterances.csv","alignments":"alignments.csv",
        "factors":[
            {"name":"phone","family":"linguistic","level":"frame","type":"categorical","source":"alignment"},
            {"name":"speaker_id","family":"paralinguistic","level":"utterance","type":"categorical","source":"speaker_id"},
            {"name":"emotion","family":"paralinguistic","level":"utterance","type":"categorical","source":"emotion"},
            {"name":"energy","family":"paralinguistic","level":"frame","type":"continuous","source":"computed:energy"},
        ]}), encoding="utf-8")
    return root


def _state(K=8, D=4):
    return {
        "sae.enc_weight": torch.randn(K,D), "sae.dec_weight": torch.randn(D,K),
        "sae.b_pre": torch.zeros(D), "routing.logits": torch.tensor([[4.,-4.]]*(K//2)+[[-4.,4.]]*(K-K//2)),
    }


class FakeSpear(nn.Module):
    def forward(self, audio, lengths):
        B, S = audio.shape; T=4; D=4
        base = audio[:, :T].unsqueeze(-1).repeat(1,1,D)
        return {"hidden_states":[base, base*.5]}


class FakeAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeSpear()


class CoreTests(unittest.TestCase):
    def test_speech_factor_metrics_compare_full_L_and_P(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            utterance_ids = np.asarray([f"u{i}" for i in range(32)])
            metadata = pd.DataFrame({
                "utterance_id": utterance_ids,
                "split": "test",
                "speaker_id": [f"s{i % 4}" for i in range(32)],
            })
            frames_per_utterance = 8
            n_frames = len(utterance_ids) * frames_per_utterance
            phones = np.tile(np.asarray(["AA", "T"] * 4), len(utterance_ids))
            speaker_codes = np.repeat((np.arange(len(utterance_ids)) % 4) + 1, frames_per_utterance)
            phone_codes = np.where(phones == "AA", 1.0, 2.0)
            # One L coordinate carries phone amplitude and one P coordinate
            # carries speaker amplitude; remaining coordinates are inactive.
            indices = np.tile(np.asarray([[0, 1, 2, 3]], dtype=np.int32), (n_frames, 1))
            values = np.stack([
                phone_codes, np.full(n_frames, .1),
                speaker_codes, np.full(n_frames, .1),
            ], axis=1).astype(np.float16)
            K = 6
            cache = FeatureCache(
                td / "features.npz", utterance_ids,
                np.arange(0, n_frames, frames_per_utterance, dtype=np.int64),
                np.full(len(utterance_ids), frames_per_utterance, dtype=np.int32),
                indices, values, phones.astype("U8"),
                np.zeros(n_frames), np.zeros(n_frames), np.zeros(n_frames),
                np.zeros((len(utterance_ids), K), dtype=np.float16),
                np.zeros((len(utterance_ids), 8), dtype=np.float16),
                np.zeros((8, 4), dtype=np.float16), np.arange(8),
                np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int16),
                np.ones(K, dtype=np.float32), K, 4,
            )
            bundle = SimpleNamespace(
                utterances=metadata,
                spec=SimpleNamespace(split_map={"test": "test"}),
            )
            out = td / "out"
            metrics, importance, repeats, summary = speech_factor_metrics(
                cache, bundle, out, max_segments=256,
                bootstrap_repetitions=8, dci_repeats=2, dci_estimators=12,
            )
            headline = summary["headline_route_contrasts"]
            self.assertGreater(headline["MIG_phone_L_minus_P"], 0.2)
            self.assertGreater(headline["MIG_speaker_P_minus_L"], 0.2)
            self.assertGreater(headline["SAP_phone_L_minus_P"], 0.2)
            self.assertGreater(headline["SAP_speaker_P_minus_L"], 0.2)
            self.assertGreater(headline["DCI_phone_informativeness_L_minus_P"], 0.4)
            self.assertGreater(headline["DCI_speaker_informativeness_P_minus_L"], 0.4)
            self.assertEqual(set(metrics["view"]), {"L", "P", "L-P", "P-L", "L|P"})
            self.assertEqual(set(metrics["scope"]), {"route_subspace"})
            self.assertEqual(
                set(metrics["capacity_mode"]), {"all_observed", "matched_active_units"},
            )
            self.assertTrue(
                ((metrics.metric == "DCI") & (metrics.component == "route_disentanglement")).any()
            )
            self.assertTrue(len(importance) > 0)
            self.assertTrue(len(repeats) > 0)
            self.assertTrue((out / "tables" / "speech_factor_metrics.csv").exists())
            self.assertTrue((out / "speech_factor_metrics.json").exists())

    def test_bundle_validation_and_checkpoint_formats(self):
        with tempfile.TemporaryDirectory() as td:
            root=_bundle(Path(td)/"bundle")
            b=AnalysisBundle(root); self.assertEqual(len(b.utterances),1)
            for key in ("model","model_state"):
                cp=Path(td)/f"{key}.pt"; torch.save({key:_state(),"analysis_config":{"topk":2,"spear_layernorm":False}},cp)
                r=load_checkpoint(cp); self.assertEqual(r.config["K"],8); self.assertTrue(r.capabilities["unit_routes"])
            cp = Path(td) / "unrouted.pt"
            torch.save({
                "model": _state(),
                "analysis_config": {"topk": 2, "spear_layernorm": False, "no_routing": True},
            }, cp)
            unrouted = load_checkpoint(cp)
            self.assertFalse(unrouted.capabilities["routes"])
            self.assertFalse(unrouted.capabilities["unit_routes"])
            self.assertFalse(unrouted.capabilities["causal"])
            route, probability = route_information(unrouted)
            np.testing.assert_array_equal(route, np.full(8, -1, dtype=np.int16))
            np.testing.assert_array_equal(probability, np.zeros(8, dtype=np.float32))
            self.assertTrue(any("one shared latent space" in warning for warning in unrouted.warnings))

    def test_missing_alignment_allows_descriptive_but_rejects_causal_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            root=_bundle(Path(td)/"bundle")
            (root/"alignments.csv").unlink()
            config=json.loads((root/"dataset.yaml").read_text()); config.pop("alignments")
            config["factors"]=[f for f in config["factors"] if f["name"]!="phone"]
            (root/"dataset.yaml").write_text(json.dumps(config))
            b=AnalysisBundle(root)
            b.require("selectivity")
            b.require("clustering")
            with self.assertRaises(AnalysisError): b.require("causal")

    def test_synthetic_factor_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            root=_bundle(Path(td)/"bundle",("test",))
            b=AnalysisBundle(root); K=4; N=100
            idx=np.tile(np.array([[0,2],[1,3]],dtype=np.int32),(N//2,1))
            val=np.ones_like(idx,dtype=np.float16)
            phones=np.array((["AA","T"]*(N//2)),dtype="U8")
            cache=FeatureCache(Path(td)/"x.npz",np.array(["u0"]),np.array([0]),np.array([N]),idx,val,
                               phones,np.zeros(N),np.linspace(-1,1,N),np.zeros(N),np.ones((1,K)),
                               np.ones((1,8)),np.ones((2,4)),np.array([0,1]),np.array([0,0,1,1]),
                               np.ones(K),K,4)
            state=_state(K,4); resolved=ResolvedModel(Path("x"),state,{"K":K,"D":4},"raw",{"unit_routes":True})
            out=Path(td)/"out";(out/"tables").mkdir(parents=True)
            health,_=health_analysis(cache,resolved,out); self.assertEqual((~health.dead).sum(),4)
            scores,profiles,_=selectivity_analysis(cache,b,out,score_splits="test")
            self.assertTrue(len(scores)>0); self.assertEqual(len(profiles),K)
            self.assertLessEqual(set(scores["factor"]), {"phone", "speaker_id"})
            self.assertIn("active_auroc_signed", scores.columns)
            unit_scores, score_summary = phone_speaker_unit_scores(cache, health, profiles, scores, out)
            self.assertIn("PhoneScore", unit_scores.columns)
            self.assertIn("SpeakerScore", unit_scores.columns)
            self.assertIn("D", unit_scores.columns)
            self.assertIn("M", unit_scores.columns)
            self.assertIn("category", unit_scores.columns)
            self.assertTrue((out/"tables"/"unit_phone_speaker_scores.csv").exists())
            self.assertIn("phone_score_formula", score_summary)
            broad_scores, _, broad_summary = selectivity_analysis(cache, b, out, factor_scope="broad", score_splits="test")
            self.assertEqual(broad_summary["factor_scope"], "broad")
            self.assertIn("energy", set(broad_scores["factor"]))
            clustered,summary=clustering_analysis(cache,profiles,out); self.assertIn("route_nmi",summary)
            geometry, geometry_summary = geometry_analysis(cache, resolved, health, out)
            self.assertIn("mean_nearest_cosine", geometry_summary)
            self.assertTrue((out/"tables"/"decoder_neighbors.csv").exists())
            self.assertEqual(len(geometry), K * min(5, K - 1))
            cache.save(); loaded=FeatureCache.load(cache.path)
            np.testing.assert_array_equal(loaded.indices,cache.indices)

    def test_phone_speaker_scores_do_not_label_zero_evidence_units(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"out"; (out/"tables").mkdir(parents=True)
            K=3
            cache=FeatureCache(Path(td)/"x.npz",np.array(["u0"]),np.array([0]),np.array([1]),
                               np.zeros((1,1),dtype=np.int32),np.zeros((1,1),dtype=np.float16),
                               np.array(["AA"],dtype="U8"),np.zeros(1),np.zeros(1),np.zeros(1),
                               np.ones((1,K)),np.ones((1,8)),np.ones((2,4)),
                               np.array([0,1]),np.array([0,0,1]),np.ones(K),K,4)
            health=pd.DataFrame({
                "unit": [0,1,2],
                "frame_frequency": [0.0,0.1,0.1],
                "utterance_frequency": [0.0,1.0,1.0],
                "observed_active": [False,True,True],
                "unobserved": [True,False,False],
                "train_like_dead": [False,False,False],
                "dead": [False,False,False],
                "mean_abs_contribution": [0.0,0.1,0.1],
            })
            profiles=pd.DataFrame({"unit": [0,1,2]})
            scores=pd.DataFrame([
                {"unit": 1, "factor": "phone", "level": "AA", "active_auroc": 0.9,
                 "active_auroc_signed": 0.8, "active_auroc_positive": 0.8,
                 "score": 0.8, "prevalence": 0.5, "q": 0.9},
                {"unit": 2, "factor": "speaker_id", "level": "s0", "active_auroc": 0.85,
                 "active_auroc_signed": 0.7, "active_auroc_positive": 0.7,
                 "amplitude_r_signed": 0.6, "amplitude_r_positive": 0.6,
                 "score": 0.6, "prevalence": 0.5, "q": 0.9},
            ])
            unit_scores, summary = phone_speaker_unit_scores(cache, health, profiles, scores, out)
            self.assertEqual(unit_scores.loc[unit_scores.unit == 0, "category"].iloc[0], "other")
            self.assertEqual(summary["phone_positive_units"], 1)
            self.assertEqual(summary["speaker_positive_units"], 1)

    def test_speaker_amplitude_metric_does_not_saturate_when_all_utts_fire(self):
        active = np.ones((6, 1), dtype=bool)
        values = np.asarray([[3.0], [2.8], [3.2], [0.2], [0.1], [0.3]])
        labels = np.asarray(["a", "a", "a", "b", "b", "b"])
        rows = _categorical_dense_scores(
            active, values, labels, "speaker_id", "paralinguistic", min_count=3,
        )
        a = next(row for row in rows if row["level"] == "a")
        self.assertEqual(a["active_auroc_positive"], 0.0)
        self.assertGreater(a["amplitude_r_positive"], 0.9)
        self.assertEqual(a["metric"], "utterance_mean_activation")

    def test_route_probabilities_and_unresolved_legacy_config(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/"legacy.pt"; torch.save({"model_state":_state()},cp)
            r=load_checkpoint(cp)
            self.assertEqual(set(unresolved_critical(r)),{"topk","spear_layernorm"})
            route,probability=route_information(r)
            np.testing.assert_array_equal(route,np.array([0,0,0,0,1,1,1,1]))
            self.assertTrue((probability>.99).all())

    def test_fixed_block_legacy_topk_fields_resolve_to_block_topk(self):
        with tempfile.TemporaryDirectory() as td:
            state = _state(K=8, D=4)
            state.pop("routing.logits")
            state["block_idx"] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
            cp = Path(td) / "fixed.pt"
            torch.save({
                "model": state,
                "analysis_config": {
                    "topk": 4,
                    "spear_layernorm": True,
                    "fixed_blocks": True,
                    "per_block_topk": True,
                    "topk_L": 3,
                    "topk_P": 1,
                    "topk_U": 0,
                },
            }, cp)
            r = load_checkpoint(cp)
            self.assertEqual(r.config["block_topk"], [3, 1, 0])
            self.assertEqual(r.config["topk_blocks"], [3, 1, 0])
            self.assertEqual(unresolved_critical(r), [])

    def test_extraction_honors_learned_route_quotas_and_global_fixed_topk(self):
        K = D = 6
        sae = SimpleNamespace(
            b_pre=torch.zeros(D),
            enc_weight=torch.eye(K),
        )
        model = SimpleNamespace(sae=sae)
        h = torch.tensor([[[9.0, 8.0, 7.0, 1.0, 0.0, -1.0]]])
        state = {
            "sae.route_topk_enabled": torch.tensor(True),
            "sae.route_topk_idx": torch.tensor([0, 0, 0, 1, 1, 1]),
            "sae.route_topk_quotas": torch.tensor([2, 1, 0]),
        }
        resolved = ResolvedModel(
            Path("x"), state, {"K": K, "D": D, "topk": 3}, "raw", {},
        )
        indices, _ = _encode_sparse(h, model, resolved)
        chosen = indices[0, 0].tolist()
        self.assertEqual(sum(i < 3 for i in chosen), 2)
        self.assertEqual(sum(i >= 3 for i in chosen), 1)

        fixed_global = ResolvedModel(
            Path("x"), {"block_idx": torch.tensor([0, 0, 0, 1, 1, 1])},
            {
                "K": K, "D": D, "topk": 3, "fixed_blocks": True,
                "per_block_topk": False, "block_topk": [2, 1, 0],
            },
            "raw", {},
        )
        self.assertIsNone(_block_spec(fixed_global))

    def test_quick_sampling_is_speaker_balanced(self):
        rows = []
        for speaker in range(12):
            for utterance in range(4):
                rows.append({
                    "speaker_id": f"s{speaker:02d}",
                    "utterance_id": f"s{speaker:02d}-u{utterance}",
                })
        sampled = _quick_sample(pd.DataFrame(rows), n=24, seed=42)
        counts = sampled["speaker_id"].value_counts()
        self.assertEqual(len(sampled), 24)
        self.assertEqual(len(counts), 8)
        self.assertTrue((counts == 3).all())

    def test_large_split_cap_is_deterministic_and_speaker_balanced(self):
        rows = pd.DataFrame([
            {"utterance_id": f"s{speaker}-{utterance}", "speaker_id": f"s{speaker}"}
            for speaker in range(10) for utterance in range(20)
        ])
        first = _speaker_balanced_sample(rows, n=50, seed=42)
        second = _speaker_balanced_sample(rows, n=50, seed=42)
        self.assertEqual(first["utterance_id"].tolist(), second["utterance_id"].tolist())
        counts = first["speaker_id"].value_counts()
        self.assertEqual(len(first), 50)
        self.assertEqual(len(counts), 10)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertEqual(
            parse_split_limits("train=3000,val=1000,test=1000"),
            {"train": 3000, "validation": 1000, "test": 1000},
        )

    def test_route_separation_reports_frozen_linear_probe(self):
        labels = np.asarray(["a"] * 10 + ["b"] * 10)
        x = np.zeros((20, 4), dtype=np.float32)
        x[:10, 0] = 1.0
        x[10:, 1] = 1.0
        metrics = _centroid_holdout_metrics(x, labels, seed=42)
        self.assertEqual(metrics["linear_probe_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        _, confusion = _holdout_metrics_and_confusion(x, labels, seed=42)
        diagonal = confusion[confusion.true_label == confusion.predicted_label]
        self.assertEqual(len(diagonal), 2)
        self.assertTrue((diagonal.row_fraction == 1.0).all())
        np.testing.assert_allclose(
            confusion.groupby("true_label").row_fraction.sum().to_numpy(), 1.0,
        )

    def test_classifier_free_pairs_control_nuisance_and_recover_geometry(self):
        labels = np.asarray(["a", "a", "b", "b"])
        nuisance = np.asarray(["s1", "s2", "s1", "s2"])
        clusters = np.asarray(["u0", "u1", "u2", "u3"])
        anchors, same, different = _controlled_pair_indices(
            labels, nuisance, clusters, seed=42,
        )
        self.assertEqual(len(anchors), 4)
        self.assertTrue(np.all(labels[anchors] == labels[same]))
        self.assertTrue(np.all(labels[anchors] != labels[different]))
        self.assertTrue(np.all(nuisance[anchors] != nuisance[same]))
        self.assertTrue(np.all(nuisance[anchors] != nuisance[different]))
        self.assertTrue(np.all(clusters[anchors] != clusters[same]))
        self.assertTrue(np.all(clusters[anchors] != clusters[different]))
        x = np.asarray([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
        ], dtype=np.float32)
        same_cosine, different_cosine = _paired_cosines(
            x, anchors, same, different,
        )
        np.testing.assert_allclose(same_cosine, 1.0)
        np.testing.assert_allclose(different_cosine, 0.0)

    def test_umap_is_finite_and_seed_reproducible(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(18, 5)).astype(np.float32)
        first = _embed_umap(x, seed=7, n_neighbors=5)
        second = _embed_umap(x, seed=7, n_neighbors=5)
        self.assertEqual(first.shape, (18, 2))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_swap_summary_keeps_modes_and_bootstrap_intervals(self):
        rows = []
        for mode_index, mode in enumerate((
            "baseline", "P_from_donor", "L_from_donor",
            "random_route_P_from_donor",
        )):
            for pair in range(8):
                rows.append({
                    "mode": mode,
                    "pair": pair,
                    "phone_recipient_accuracy": .9 - .1 * mode_index + .01 * pair,
                    "donor_speaker_match": float(mode == "P_from_donor"),
                    "recipient_speaker_match": float(mode == "baseline"),
                    "donor_speaker_probability": .1 + .2 * mode_index,
                    "recipient_speaker_probability": .8 - .2 * mode_index,
                    "reconstruction_shift_mse": .01 * mode_index,
                })
        summary = _bootstrap_mode_summary(
            pd.DataFrame(rows), seed=42, repetitions=100,
        )
        self.assertEqual(
            summary["mode"].tolist(),
            ["baseline", "P_from_donor", "L_from_donor", "random_route_P_from_donor"],
        )
        self.assertTrue((summary["pairs"] == 8).all())
        self.assertTrue(
            (summary["phone_recipient_accuracy_ci95_low"]
             <= summary["phone_recipient_accuracy"]).all()
        )
        self.assertTrue(
            (summary["phone_recipient_accuracy"]
             <= summary["phone_recipient_accuracy_ci95_high"]).all()
        )

    def test_phone_confusion_selects_positive_unique_diagonal_units(self):
        with tempfile.TemporaryDirectory() as td:
            root = _bundle(Path(td) / "bundle", ("train", "val", "test"))
            bundle = AnalysisBundle(root)
            K = 4
            phones = np.asarray(["AA", "AA", "T", "T"] * 3, dtype="U8")
            indices = np.asarray([[0], [0], [1], [1]] * 3, dtype=np.int32)
            cache = FeatureCache(
                Path(td) / "x.npz",
                np.asarray(["u0", "u1", "u2"]),
                np.asarray([0, 4, 8]), np.asarray([4, 4, 4]),
                indices, np.ones_like(indices, dtype=np.float16), phones,
                np.zeros(12), np.zeros(12), np.zeros(12),
                np.ones((3, K)), np.ones((3, 8)), np.ones((3, 4)),
                np.asarray([0, 4, 8]), np.asarray([0, 0, 1, 1]),
                np.ones(K), K, 4,
            )
            out = Path(td) / "out"; (out / "tables").mkdir(parents=True)
            matrix, selected, summary = phone_unit_confusion(
                cache, bundle, out, min_phone_frames=1,
                selection_splits="train,validation", evaluation_splits="test",
            )
            self.assertEqual(len(selected), 2)
            self.assertEqual(selected["unit"].nunique(), 2)
            self.assertTrue(selected["evaluation_diagonal_is_max"].all())
            self.assertIn("phone_family", selected.columns)
            self.assertTrue((selected["evaluation_margin"] == 1.0).all())
            self.assertTrue((selected["evaluation_max_other_probability"] == 0.0).all())
            self.assertTrue((selected["selection_margin"] > 0).all())
            self.assertEqual(summary["evaluation_diagonal_max_fraction"], 1.0)
            for _, row in matrix.iterrows():
                self.assertEqual(float(row[row.selected_phone]), 1.0)
            cache.route[:] = -1
            unrouted_matrix, unrouted_selected, unrouted_summary = phone_unit_confusion(
                cache, bundle, out, min_phone_frames=1,
                selection_splits="train,validation", evaluation_splits="test",
            )
            self.assertEqual(len(unrouted_selected), 2)
            self.assertEqual(set(unrouted_selected["route"]), {"unassigned"})
            self.assertEqual(unrouted_summary["selected_units"], 2)
            self.assertEqual(len(unrouted_matrix), 2)

    def test_fake_spear_cli_vertical_slice(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); root=_bundle(td/"bundle")
            cp=td/"model.pt"; torch.save({"model":_state(),"analysis_config":{"topk":2,"spear_layernorm":False}},cp)
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoModel = FakeAutoModel
            # patch.dict restores the entire module registry on exit. Import
            # UMAP first so its lazily loaded NumPy/Numba extensions are not
            # removed and then loaded a second time later in this process.
            _load_umap()
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                result=run_analysis(
                    cp, root, "health,atlas,selectivity,clustering,similarity,geometry",
                    output_dir=td/"result", device="cpu", profile="quick",
                    score_splits="test")
            self.assertTrue(result.artifacts["report"].exists())
            self.assertTrue((td/"result"/"tables"/"units.csv").exists())
            self.assertTrue((td/"result"/"tables"/"unit_phone_speaker_scores.csv").exists())
            self.assertTrue((td/"result"/"tables"/"decoder_neighbors.csv").exists())
            self.assertTrue((td/"result"/"report"/"index.html").exists())
            self.assertFalse((td/"result"/"report"/"assets"/"spectrograms").exists())
            self.assertFalse((td/"result"/"report"/"assets"/"traces").exists())
            unit_pages = list((td/"result"/"report"/"units").glob("*.html"))
            self.assertFalse(unit_pages)
            quick_html = (td/"result"/"report"/"index.html").read_text()
            self.assertIn("Unit table", quick_html)
            self.assertIn("Quick smoke test", quick_html)
            self.assertIn("only units observed", quick_html)
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                rich=run_analysis(
                    cp, root, "health,atlas",
                    output_dir=td/"rich_result", device="cpu", profile="quick",
                    atlas_assets="traces")
            self.assertTrue(rich.artifacts["report"].exists())
            rich_pages = list((td/"rich_result"/"report"/"units").glob("*.html"))
            self.assertTrue(rich_pages)
            self.assertNotIn("<audio", rich_pages[0].read_text())
            self.assertTrue((td/"rich_result"/"report"/"assets"/"traces").exists())

    def test_unrouted_selectivity_report_avoids_route_claims(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); root = _bundle(td / "bundle")
            cp = td / "unrouted.pt"
            torch.save({
                "model": _state(),
                "analysis_config": {"topk": 2, "spear_layernorm": False, "no_routing": True},
            }, cp)
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoModel = FakeAutoModel
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                result = run_analysis(
                    cp, root, "health,selectivity", output_dir=td / "result",
                    device="cpu", profile="quick", score_splits="test",
                )
            report = (td / "result" / "report" / "index.html").read_text()
            self.assertIn("Unrouted SAE Unit Analysis", report)
            self.assertIn("one unrouted SAE representation", report)
            self.assertNotIn("Disentanglement summary by route", report)
            self.assertNotIn("Probing-style route-vector separation", report)
            self.assertTrue((td / "result" / "tables" / "unrouted_unit_summary.csv").exists())
            self.assertNotIn("factor_metrics", result.completed)

    def test_timit_style_metadata_counts_as_l_route_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            (out / "tables").mkdir(parents=True)
            profiles = pd.DataFrame([
                {
                    "unit": 0, "route": "L", "route_id": 0,
                    "linguistic_score": 0.5, "paralinguistic_score": 9.0,
                    "sex__score": 9.0, "speaker_id__score": 0.0,
                },
                {
                    "unit": 1, "route": "L", "route_id": 0,
                    "linguistic_score": 8.0, "paralinguistic_score": 0.0,
                    "phone__score": 8.0, "speaker_id__score": 0.0,
                },
                {
                    "unit": 2, "route": "P", "route_id": 1,
                    "linguistic_score": 7.0, "paralinguistic_score": 0.0,
                    "phone__score": 7.0, "speaker_id__score": 0.0,
                },
                {
                    "unit": 3, "route": "P", "route_id": 1,
                    "linguistic_score": 0.0, "paralinguistic_score": 7.0,
                    "phone__score": 0.0, "speaker_id__score": 7.0,
                },
            ])
            scores = pd.DataFrame([
                {"unit": 0, "factor": "sex", "family": "paralinguistic", "score": 9.0, "q": 0.001},
                {"unit": 1, "factor": "phone", "family": "linguistic", "score": 8.0, "q": 0.001},
                {"unit": 2, "factor": "phone", "family": "linguistic", "score": 7.0, "q": 0.001},
                {"unit": 3, "factor": "speaker_id", "family": "paralinguistic", "score": 7.0, "q": 0.001},
            ])
            units, leaky, route_summary, summary = disentanglement_tables(
                None, profiles, scores, out, focus="broad",
            )
            unit0 = units[units.unit == 0].iloc[0]
            self.assertTrue(bool(unit0.route_violation))
            self.assertIn("metadata_in_L", unit0.issue_tags)
            self.assertGreater(float(unit0.leakage_score), 0.0)
            self.assertGreater(summary["thesis_summary"]["L_paralinguistic_leak_fraction"], 0.0)
            self.assertEqual(int(leaky.iloc[0].unit), 0)
            self.assertIn("metadata_paralinguistic_selective_fraction", route_summary.columns)

    def test_categorical_headlines_use_effect_size_not_frame_q(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            (out / "tables").mkdir(parents=True)
            profiles = pd.DataFrame([
                {"unit": 0, "route": "L", "route_id": 0,
                 "linguistic_score": 0.0, "paralinguistic_score": 0.2,
                 "SpeakerScore": 0.2},
                {"unit": 1, "route": "L", "route_id": 0,
                 "linguistic_score": 0.0, "paralinguistic_score": 0.01,
                 "SpeakerScore": 0.01},
            ])
            scores = pd.DataFrame([
                {"unit": 0, "factor": "speaker_id", "family": "paralinguistic",
                 "score": 0.2, "q": 1.0, "active_auroc_positive": 0.2},
                {"unit": 1, "factor": "speaker_id", "family": "paralinguistic",
                 "score": 100.0, "q": 0.0, "active_auroc_positive": 0.01},
            ])
            units, _, _, summary = disentanglement_tables(
                None, profiles, scores, out, focus="speaker_content",
            )
            selected = units.set_index("unit")["speaker_selective"].to_dict()
            self.assertTrue(bool(selected[0]))
            self.assertFalse(bool(selected[1]))
            self.assertEqual(
                summary["categorical_selection"],
                "positive_phone_auroc_or_utterance_amplitude_r",
            )

    def test_build_timit_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "TIMIT"
            spk = root / "TRAIN" / "DR1" / "FCJF0"
            spk.mkdir(parents=True)
            _wav(spk / "SA1.WAV")
            (spk / "SA1.TXT").write_text("0 16000 she had your dark suit\n")
            (spk / "SA1.PHN").write_text("0 8000 sh\n8000 16000 iy\n")

            out = Path(td) / "bundle"
            build_bundle(root, out)
            bundle = AnalysisBundle(out)
            self.assertEqual(len(bundle.utterances), 1)
            self.assertEqual(len(bundle.alignments), 2)
            self.assertTrue((out / "dataset.yaml").exists())
            self.assertTrue((out / "audio" / "TRAIN" / "DR1" / "FCJF0" / "SA1.WAV").exists())
            self.assertIn("phone", {factor.name for factor in bundle.spec.factors})
            self.assertNotIn("sex", {factor.name for factor in bundle.spec.factors})

    def test_build_librispeech_bundle_without_phone_alignments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "LibriSpeech"
            for split, uid in (("train-clean-100", "1-2-0000"), ("dev-clean", "3-4-0000"), ("test-clean", "5-6-0000")):
                speaker, chapter, _ = uid.split("-")
                folder = root / split / speaker / chapter
                folder.mkdir(parents=True)
                _wav(folder / f"{uid}.wav")
                (folder / f"{speaker}-{chapter}.trans.txt").write_text(f"{uid} a small test\n")

            out = Path(td) / "libri_bundle"
            build_librispeech_bundle(root, out, max_train=1, max_validation=1, max_test=1)
            bundle = AnalysisBundle(out)
            self.assertEqual(len(bundle.utterances), 3)
            self.assertIsNone(bundle.alignments)
            self.assertIn("speaker_id", {factor.name for factor in bundle.spec.factors})
            bundle.require("selectivity")
            bundle.require("clustering")
            with self.assertRaises(AnalysisError):
                bundle.require("causal")

    def test_librispeech_mfa_bridge_imports_phone_alignments(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            root = td / "LibriSpeech"
            for split, uid in (("train-clean-100", "1-2-0000"), ("dev-clean", "3-4-0000"), ("test-clean", "5-6-0000")):
                speaker, chapter, _ = uid.split("-")
                folder = root / split / speaker / chapter
                folder.mkdir(parents=True)
                _wav(folder / f"{uid}.wav")
                (folder / f"{speaker}-{chapter}.trans.txt").write_text(f"{uid} A SMALL TEST\n")

            bundle_root = td / "libri_bundle"
            build_librispeech_bundle(root, bundle_root, max_train=1, max_validation=1, max_test=1)

            mfa_corpus = td / "mfa_corpus"
            prepare_corpus(bundle_root, mfa_corpus)
            self.assertTrue((mfa_corpus / "audio" / "1" / "1-2-0000.lab").exists())
            self.assertTrue((mfa_corpus / "mfa_utterance_map.csv").exists())

            aligned = td / "aligned"
            aligned.mkdir()
            (aligned / "1-2-0000.TextGrid").write_text(
                '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.2
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.05
            text = "sil"
        intervals [2]:
            xmin = 0.05
            xmax = 0.12
            text = "AH0"
        intervals [3]:
            xmin = 0.12
            xmax = 0.2
            text = "T"
''',
                encoding="utf-8",
            )
            intervals = parse_textgrid(aligned / "1-2-0000.TextGrid")
            self.assertEqual(len(intervals), 3)

            out = td / "aligned_bundle"
            import_alignments(
                bundle_root,
                aligned,
                output=out,
                utterance_map=mfa_corpus / "mfa_utterance_map.csv",
                min_coverage=0.0,
            )
            aligned_bundle = AnalysisBundle(out)
            self.assertIn("phone", {factor.name for factor in aligned_bundle.spec.factors})
            self.assertEqual(len(aligned_bundle.alignments), 2)
            self.assertEqual(set(aligned_bundle.alignments["phone"]), {"AH", "T"})
            aligned_bundle.require("causal")


if __name__ == "__main__":
    unittest.main()
