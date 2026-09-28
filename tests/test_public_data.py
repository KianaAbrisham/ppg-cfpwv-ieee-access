"""Verify subject identity and source integrity in the PWDB converter."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pwdb_converter", ROOT / "tools" / "prepare_public_pwdb.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class PublicDataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = np.arange(1, 4375)

    def test_shuffling_rows_preserves_subject_identity(self):
        frame = pd.DataFrame(
            {"Subject Number": self.ids, " value ": self.ids * 2}
        ).sample(frac=1, random_state=19)
        path = self.root / "subjects.csv"
        frame.to_csv(path, index=False)
        loaded = converter.table(path)
        np.testing.assert_array_equal(loaded.index, self.ids)
        np.testing.assert_array_equal(loaded["value"], self.ids * 2)

    def test_duplicate_ids_rejected(self):
        ids = self.ids.copy()
        ids[-1] = ids[0]
        path = self.root / "subjects.csv"
        pd.DataFrame({"Subject Number": ids}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate IDs"):
            converter.table(path)

    def test_missing_subject_rejected(self):
        path = self.root / "subjects.csv"
        pd.DataFrame({"Subject Number": self.ids[:-1]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Unexpected subject set"):
            converter.table(path)

    def test_corrupt_download_rejected(self):
        (self.root / "PWs_csv.zip").write_bytes(b"not the official archive")
        with self.assertRaisesRegex(ValueError, "Official MD5 mismatch"):
            converter.obtain(self.root, False)


if __name__ == "__main__":
    unittest.main()
