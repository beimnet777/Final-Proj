from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from SAEUnitAnalysis.analyses import (
    clustering_analysis,
    disentanglement_tables,
    health_analysis,
    selectivity_analysis,
)
from SAEUnitAnalysis.bundle import AnalysisBundle
from SAEUnitAnalysis.build_librispeech_bundle import build_bundle as build_librispeech_bundle
from SAEUnitAnalysis.build_timit_bundle import build_bundle
from SAEUnitAnalysis.checkpoint import load_checkpoint, route_information, unresolved_critical
from SAEUnitAnalysis.extraction import FeatureCache
from SAEUnitAnalysis.pipeline import run_analysis
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
    def test_bundle_validation_and_checkpoint_formats(self):
        with tempfile.TemporaryDirectory() as td:
            root=_bundle(Path(td)/"bundle")
            b=AnalysisBundle(root); self.assertEqual(len(b.utterances),1)
            for key in ("model","model_state"):
                cp=Path(td)/f"{key}.pt"; torch.save({key:_state(),"analysis_config":{"topk":2,"spear_layernorm":False}},cp)
                r=load_checkpoint(cp); self.assertEqual(r.config["K"],8); self.assertTrue(r.capabilities["unit_routes"])

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
            scores,profiles,_=selectivity_analysis(cache,b,out)
            self.assertTrue(len(scores)>0); self.assertEqual(len(profiles),K)
            clustered,summary=clustering_analysis(cache,profiles,out); self.assertIn("route_nmi",summary)
            cache.save(); loaded=FeatureCache.load(cache.path)
            np.testing.assert_array_equal(loaded.indices,cache.indices)

    def test_route_probabilities_and_unresolved_legacy_config(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/"legacy.pt"; torch.save({"model_state":_state()},cp)
            r=load_checkpoint(cp)
            self.assertEqual(set(unresolved_critical(r)),{"topk","spear_layernorm"})
            route,probability=route_information(r)
            np.testing.assert_array_equal(route,np.array([0,0,0,0,1,1,1,1]))
            self.assertTrue((probability>.99).all())

    def test_fake_spear_cli_vertical_slice(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); root=_bundle(td/"bundle")
            cp=td/"model.pt"; torch.save({"model":_state(),"analysis_config":{"topk":2,"spear_layernorm":False}},cp)
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoModel = FakeAutoModel
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                result=run_analysis(cp,root,"health,atlas",output_dir=td/"result",device="cpu",profile="quick")
            self.assertTrue(result.artifacts["report"].exists())
            self.assertTrue((td/"result"/"tables"/"units.csv").exists())

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
            units, leaky, route_summary, summary = disentanglement_tables(None, profiles, scores, out)
            unit0 = units[units.unit == 0].iloc[0]
            self.assertTrue(bool(unit0.route_violation))
            self.assertIn("metadata_in_L", unit0.issue_tags)
            self.assertGreater(float(unit0.leakage_score), 0.0)
            self.assertGreater(summary["thesis_summary"]["L_paralinguistic_leak_fraction"], 0.0)
            self.assertEqual(int(leaky.iloc[0].unit), 0)
            self.assertIn("metadata_paralinguistic_selective_fraction", route_summary.columns)

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
            self.assertIn("sex", {factor.name for factor in bundle.spec.factors})

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


if __name__ == "__main__":
    unittest.main()
