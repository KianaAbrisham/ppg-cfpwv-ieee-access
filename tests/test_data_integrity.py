from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import paper_code as paper

class IntegrityTests(unittest.TestCase):
    def test_explicit_unique_ids_required(self):
        for frame in [pd.DataFrame({'Age': [40]}),
                      pd.DataFrame({'Subject Number': [1, 1]}),
                      pd.DataFrame({'Subject Number': [1.5]}),
                      pd.DataFrame({'Subject Number': [np.nan]})]:
            with self.subTest(frame=frame.to_dict()), self.assertRaises((ValueError, KeyError)):
                paper.validate_subject_ids(frame, 'fixture')

    def test_waveform_keeps_first_sample_and_trims_only_trailing_padding(self):
        row = pd.Series({'Unnamed: 0': 0, 'Subject Number': 1,
                         's0': 0.1, 's1': 0.5, 's2': 1.0, 's3': 0.4, 's4': 0.2, 's5': np.nan})
        np.testing.assert_allclose(paper.extract_waveform(row), [0.1, 0.5, 1., 0.4, 0.2])
        row['s2'] = np.nan
        with self.assertRaises(ValueError):
            paper.extract_waveform(row)

    def test_target_alignment_uses_ids_not_position(self):
        features = pd.DataFrame({'Subject Number': [2, 1], 'feature': [8., 9.]})
        targets = pd.DataFrame({'Subject Number': [1, 2], 'PWV_cf [m/s]': [5., 7.]})
        aligned = paper.prepare_regression_data(features, targets)
        np.testing.assert_array_equal(aligned['cf_pwv'], [7., 5.])
        with self.assertRaises(ValueError):
            paper.prepare_regression_data(features, pd.concat([targets, targets.iloc[[0]]]))
        with self.assertRaises(ValueError):
            paper.prepare_regression_data(features, targets.iloc[[0]])
        targets.loc[0, 'PWV_cf [m/s]'] = -1
        with self.assertRaises(ValueError):
            paper.prepare_regression_data(features, targets)

    def test_feature_extraction_records_invalid_waveform(self):
        samples = np.array([0., 0.2, 0.6, 1., 0.7, 0.4, 0.2, 0.1, 0.])
        waves = pd.DataFrame([samples, samples], columns=[f's{i}' for i in range(len(samples))])
        waves.insert(0, 'Subject Number', [1, 2])
        waves.loc[1, 's2'] = np.nan
        row = {'Subject Number': 1, 'Age': 40, 'Radial_SI': 5., 'Radial_RI': 0.3}
        for point, time, value in [('sys', 0.006, 1.), ('dia', 0.012, 0.2),
                                   ('a', 0.002, 1.), ('b', 0.004, -0.5),
                                   ('c', 0.006, 0.1), ('d', 0.008, -0.2), ('e', 0.010, 0.1)]:
            row[f'Radial_PPG{point}_T'] = time
            row[f'Radial_PPG{point}_V'] = value
        indices = pd.DataFrame([row, {**row, 'Subject Number': 2}])
        features = paper.build_feature_table(waves, indices, 'Radial')
        self.assertEqual(features['Subject Number'].tolist(), [1])
        self.assertEqual(features.attrs['exclusions'][0]['Subject Number'], 2)
        self.assertIn('gaps', features.attrs['exclusions'][0]['Reason'])
        self.assertAlmostEqual(features[paper.AMP_COL].iloc[0], 0.5)
        self.assertEqual(paper.map_age(features, indices)['Age'].tolist(), [40])

    def test_synthetic_five_fold_training_preserves_subject_ids(self):
        # A software execution check, not a paper result. Avoid writing high-resolution figures.
        rng = np.random.default_rng(42)
        values = rng.normal(size=(30, 3))
        features = pd.DataFrame(values, columns=['one', 'two', 'three'])
        features.insert(0, 'Subject Number', np.arange(1, 31))
        targets = pd.DataFrame({'Subject Number': np.arange(1, 31),
                                'PWV_cf [m/s]': 7 + 0.5*values[:, 0] + 0.1*rng.normal(size=30)})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            targets.to_csv(directory/'targets.csv', index=False)
            with patch.object(paper, 'TAB_OUT_DIR', directory/'tables'), \
                 patch.object(paper, 'FIG_OUT_DIR', directory/'figures'), \
                 patch('matplotlib.figure.Figure.savefig'), patch.object(paper.plt, 'show'):
                paper.train_cf_pwv_model(features, directory/'targets.csv')
            saved = pd.read_csv(directory/'tables/out_of_fold_predictions.csv')
            self.assertEqual(len(saved), 30)
            self.assertEqual(saved['Subject Number'].nunique(), 30)
            self.assertEqual(set(saved['Fold']), {1, 2, 3, 4, 5})
            self.assertTrue(np.isfinite(saved['Predicted_cf_pwv_m_s']).all())

if __name__ == '__main__':
    unittest.main()
