import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from mmfnd.ablation.fusion import Fusion, dtf
from mmfnd.ablation.metrics import metrics, choose_review_fraction, holm, paired, bootstrap_metrics
from mmfnd.ablation.report import validate, generate


def frame(split="val"):
    return pd.DataFrame({"dataset":"unit", "seed":11, "split":split,
        "id":[split+str(i) for i in range(8)], "label":[0,1,0,1,0,1,0,1],
        "early_logit":[-2,2,-1,1,2,-2,1,-1], "hier_logit":[-2,2,-1,1,2,-2,1,-1],
        "dtf":0., "domain_n":0., "p_sup":0., "p_ref":0., "p_nei":1.,
        "s_max":0., "evidence_available":0})


class AblationTests(unittest.TestCase):
    def test_report_artifacts_from_unit_fixture(self):
        # Synthetic fixtures stay in a temporary test directory, never paper outputs.
        scratch = Path("outputs/test_scratch").resolve()
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            self.assertTrue(root.resolve().is_relative_to(scratch))
            data = pd.concat([pd.concat([frame(),frame("test")]).assign(seed=s) for s in (11,22)])
            data.to_csv(root/"scores.csv",index=False)
            generate(root/"scores.csv",root/"report",draws=10)
            summary = pd.read_csv(root/"report/metrics_summary.csv")
            self.assertEqual(len(summary),7)
            for name in ("unit_table.tex","unit_component_gain.pdf","unit_reliability.png",
                         "unit_score_stress.svg","mcnemar.csv","paired_bootstrap.csv"):
                self.assertTrue((root/"report"/name).is_file())

    def test_fast_bootstrap_matches_reference_with_ties(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0,2,100)
        p = np.round(rng.uniform(.01,.99,100),1)
        reference = metrics(y,p)
        for key,value in bootstrap_metrics(y,p).items():
            self.assertAlmostEqual(value,reference[key])

    def test_no_context_exact_fallback(self):
        val, test = frame(), frame("test")
        expected = Fusion("A1").fit(val).predict(test)
        for config in ("A2","A3","A4"):
            np.testing.assert_allclose(Fusion(config).fit(val).predict(test), expected)
        np.testing.assert_allclose(Fusion("A5").fit(val).predict(test),Fusion("A6").fit(val).predict(test))

    def test_validation_only(self):
        with self.assertRaises(ValueError):
            Fusion("A6").fit(frame("test"))
        model = Fusion("A6").fit(frame())
        a = frame("test")
        b = a.copy()
        b.label = 1-b.label
        np.testing.assert_array_equal(model.predict(a),model.predict(b))

    def test_gate_rejects_low_quality_extreme_context(self):
        val, test = frame(),frame("test")
        model = Fusion("A6").fit(val)
        test["evidence_available"] = 1
        test["p_sup"],test["p_ref"],test["p_nei"] = 1e-5,0.,1-1e-5
        test["s_max"] = -1.
        np.testing.assert_allclose(model.predict(test),model.predict(frame("test")),atol=1e-4)

    def test_dtf_sign_and_unseen(self):
        self.assertEqual(float(dtf(0,0)),0.)
        self.assertGreater(float(dtf(20,1)),0.)
        self.assertLess(float(dtf(1,20)),0.)
        with self.assertRaises(ValueError):
            dtf(-1,0)

    def test_cost_small_penalties_no_review(self):
        y = np.array([0,0,1,1])
        p = np.array([.9,.8,.2,.1])
        for penalty in (.1,.5,1.):
            self.assertEqual(choose_review_fraction(y,p,penalty),0.)

    def test_fake_positive_metrics(self):
        m = metrics(np.array([0,0,1,1]),np.array([.1,.8,.9,.2]))
        self.assertEqual(m["fnr"],.5)
        self.assertEqual(m["fpr"],.5)
        self.assertEqual(m["macro_f1"],.5)

    def test_schema_rejects_overlap_and_bad_probabilities(self):
        good = pd.concat([frame(),frame("test")],ignore_index=True)
        validate(good)
        bad = good.copy()
        bad.loc[8,"id"] = "val0"
        with self.assertRaises(ValueError):
            validate(bad)
        bad = good.copy()
        bad.loc[0,"p_nei"] = .3
        with self.assertRaises(ValueError):
            validate(bad)

    def test_identical_predictions_zero_effect(self):
        y,p = np.array([0,1,0,1]),np.array([.1,.9,.7,.2])
        rows, mc = paired(y,p,p,draws=10)
        self.assertEqual(mc["p_value"],1.)
        for row in rows:
            self.assertEqual(row["difference"],0.)
            self.assertEqual(row["ci_low"],0.)
            self.assertEqual(row["ci_high"],0.)
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])


if __name__ == "__main__":
    unittest.main()
