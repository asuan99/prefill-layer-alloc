"""energy.csv layout regression (review 2026-10-12, D1): job_entry >= 52a3b97 writes 3 columns
(epoch_s, mono_s, energy_mJ); every reader must accept both the 2- and the 3-column layout."""
import importlib.util
import os
import sys
import tempfile
import unittest

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ENG, "results", "power_cap")


def _load(path, name):
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


class TestEnergyLayouts(unittest.TestCase):
    ROWS = [(1791771464.1, 100.0, 5000), (1791771464.2, 100.1, 5030), (1791771464.3, 100.2, 5061)]

    def _write(self, three):
        fd, p = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as fh:
            fh.write("epoch_s,mono_s,energy_mJ\n" if three else "epoch_s,energy_mJ\n")
            for t, m, e in self.ROWS:
                fh.write(f"{t:.6f},{m:.6f},{e}\n" if three else f"{t:.3f},{e}\n")
        self.addCleanup(os.remove, p)
        return p

    def test_n2_decode_power_reader(self):
        m = _load(os.path.join(PC, "n2_decode_power", "pc_n2_decode_power.py"), "_pc_n2dp")
        for three in (False, True):
            rows = m.read_energy(self._write(three))
            self.assertEqual([e for _, e in rows], [5000, 5030, 5061], three)

    def test_pc_common_reader(self):
        m = _load(os.path.join(PC, "analysis_2026-10-07", "pc_common.py"), "_pc_common")
        for three in (False, True):
            from pathlib import Path
            rows = m.parse_energy(Path(self._write(three)))
            self.assertEqual([e for _, e in rows], [5000, 5030, 5061], three)


if __name__ == "__main__":
    unittest.main()
