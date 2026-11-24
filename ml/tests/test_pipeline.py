"""Testes rápidos do pacote ml/csi (sem TensorFlow).  python -m unittest discover -s ml/tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csi import presence as P  # noqa: E402
from csi.labels import LABELS, label_times, load_manifest, sessions  # noqa: E402
from csi.logs import motion_state_per_window, parse_frame_states, parse_signatures  # noqa: E402
from csi.motion import CORE7, log_ratio, setups, stack  # noqa: E402
from csi.nets import cell_balanced_weights, train_binary, predict_proba  # noqa: E402
from csi.paths import DATA  # noqa: E402
from csi.temporal import EMPTY, MOVING, STATIC, TemporalFilter, fuse_states, run_filter  # noqa: E402

SIG = ('I (2558) CSI_SIG: v=1 window=1 t_ms=2214 rssi=-59.48 raw_med=20.9486 clip=0 q12='
       + '1000' * 51)


class Logs(unittest.TestCase):
    def test_signature_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.log'
            p.write_text(SIG + '\n' + SIG.replace('window=1 t_ms=2214', 'window=2 t_ms=2700') + '\n'
                         + SIG[:-10] + '\n')  # linha truncada é ignorada
            df = parse_signatures(p)
            self.assertEqual(len(df), 2)
            self.assertAlmostEqual(df['q00'].iloc[0], 1.0)
            self.assertAlmostEqual(df['t_s'].iloc[1], 2.7)

    def test_motion_state_steps(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.log'
            p.write_text('I (1) CSI_FRAME: Unified diagnostics: every_window=0\n'
                         'I (2) CSI_FRAME: window=5 f9_ratio=1 core7_p=0.9 stable=MOVING logistic_stable=MOVING\n'
                         'I (3) CSI_FRAME: window=9 f9_ratio=1 core7_p=0.1 stable=NOT_MOVING logistic_stable=NOT_MOVING\n')
            fr = parse_frame_states(p)
            st = motion_state_per_window(fr, np.arange(1, 12), 'mlp_stable')
            self.assertEqual(st.tolist(), [False] * 4 + [True] * 4 + [False] * 3)


class Labels(unittest.TestCase):
    def test_manifest_files_exist(self):
        for s in sessions():
            self.assertTrue(s.path.exists(), s.path)
            for _, lab, a, b in s.phases:
                self.assertIn(lab, LABELS + ('UNKNOWN',))
                self.assertLess(a, b)
        self.assertEqual(len(sessions('dataset')), 6)

    def test_guards(self):
        t = np.arange(0, 30, 0.5)
        lab, ph, sc = label_times(t, [('E', 'EMPTY', 0, 10), ('S', 'STATIC', 10, 30)], 7, 5)
        self.assertFalse(sc[t < 10].any())   # fase de 10 s some com guardas 7 + 5
        self.assertFalse(sc[t == 12].any())   # dentro da guarda inicial de S
        self.assertTrue(sc[t == 20].all())
        self.assertFalse(sc[t == 26].any())   # guarda final


class Presence(unittest.TestCase):
    def test_identical_profile_gives_zero_shape_features(self):
        rng = np.random.default_rng(0)
        q = 1 + 0.2 * rng.random((30, 51))
        p = P.shape_profiles(q)
        f = P.raw_features(p[:1], p[:1], np.array([-60.0]), p[0], -60.0)
        np.testing.assert_allclose(f, 0, atol=1e-12)

    def test_gain_invariance(self):
        rng = np.random.default_rng(1)
        q = 1 + 0.2 * rng.random((5, 51))
        np.testing.assert_allclose(P.shape_profiles(q), P.shape_profiles(3.7 * q))

    def test_lag_uses_oldest_available(self):
        p = np.arange(8)[:, None] * np.ones((8, 51))
        lag = P.lagged(p, 5)
        self.assertEqual(lag[:, 0].tolist(), [0, 0, 0, 0, 0, 0, 1, 2])

    def test_session_table_excludes_calibration(self):
        s = [x for x in sessions('dataset') if x.id == 'C'][0]
        tb = P.session_table(s)
        self.assertEqual(int(tb['calibration'].sum()), P.CAL_WINDOWS)
        self.assertFalse((tb['scored'] & tb['calibration']).any())
        self.assertTrue(np.isfinite(tb[P.X_COLS].to_numpy()).all())


class Motion(unittest.TestCase):
    def test_setups_and_transform(self):
        sets = setups('train')
        self.assertEqual(len(sets), 5)
        x, y, cells, _, _ = stack(sets)
        self.assertEqual(x.shape[1], 7)
        np.testing.assert_allclose(log_ratio(np.ones(7), np.ones(7)), 0)
        self.assertEqual(CORE7, (3, 4, 5, 7, 8, 9, 10))
        w = cell_balanced_weights(cells, y)
        self.assertAlmostEqual(w[y == 1].sum(), w[y == 0].sum(), places=6)


class Temporal(unittest.TestCase):
    def test_filter(self):
        f = TemporalFilter()
        out = [f.observe(v)[1] for v in [0.9] * 5 + [0.5] + [0.0] * 5]
        self.assertEqual(out[4], True)
        self.assertEqual(out[5], True)    # histerese
        self.assertEqual(out[-1], False)
        act, _ = run_filter(np.array([0.9] * 4))
        self.assertFalse(act.any())       # aquecimento

    def test_fusion(self):
        st = fuse_states(np.array([0, 1, 1, 0], bool), np.array([0, 0, 1, 1], bool))
        self.assertEqual(st.tolist(), [EMPTY, STATIC, MOVING, MOVING])


class Training(unittest.TestCase):
    def test_deterministic(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(200, 7)).astype(np.float32)
        y = (z[:, 0] > 0).astype(int)
        w = np.ones(200)
        a = predict_proba(train_binary(z, y, w, epochs=30), z)
        b = predict_proba(train_binary(z, y, w, epochs=30), z)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
