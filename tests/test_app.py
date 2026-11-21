from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_node_changes_and_impossible_demand(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=60).run()
        self.assertFalse(app.exception)
        def gap_data():
            return next(t.value for t in app.dataframe if 'max_minus_min' in t.value.columns and t.value.m.nunique() == 1)
        before = gap_data()
        app.number_input[2].set_value(2000).run(timeout=60)
        after = gap_data()
        self.assertEqual(set(after.m), {2000})
        self.assertNotEqual(before.loc[before.model == 1, 'max_minus_min'].max(),
                            after.loc[after.model == 1, 'max_minus_min'].max())
        for n in [3,6]:
            app.number_input[0].set_value(n).run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(len(next(t.value for t in app.dataframe if 'Model 2 pairs' in t.value.columns)),n-1)
        app.number_input[2].set_value(1).run(timeout=60)
        self.assertFalse(app.exception)
        path_table=next(t.value for t in app.dataframe if 'F_FI' in t.value.columns)
        self.assertTrue((path_table['F_FI']==.5).all())

    def test_switch_to_star_and_recalculate(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=60).run()
        app.selectbox[0].set_value('Star').run(timeout=60)
        self.assertFalse(app.exception)
        figure_titles = [h.value for h in app.subheader if h.value.startswith('Figure ')]
        self.assertEqual([h.split(' · ')[0] for h in figure_titles],
                         ['Figure 8', 'Figure 9', 'Figure 11'])
        self.assertTrue(all('Star' in h for h in figure_titles))
        waiting = next(t.value for t in app.dataframe if 'max_minus_min' in t.value.columns and set(t.value.m) == {50})
        self.assertEqual(set(waiting.model), {0,1,2})
        self.assertGreater(waiting.loc[waiting.model == 1, 'best'].max(), .5)
        self.assertGreater(waiting.loc[waiting.model == 2, 'best'].max(), .5)
        comparison_modes = next(w for w in app.number_input if w.label == 'Star model-comparison modes')
        comparison_modes.set_value(65).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any('max_minus_min' in t.value.columns and set(t.value.m) == {65}
                            for t in app.dataframe))
        table = next(t.value for t in app.dataframe if 'Model 2 pairs' in t.value.columns)
        self.assertEqual(table['Edge'].tolist(), ['0–1','0–2','0–3','0–4'])
        self.assertEqual(table['Model 2 pairs'].tolist(), [4]*4)
        self.assertEqual(app.number_input[1].value, 20.)
        self.assertEqual(app.number_input[2].value, 10)
        app.number_input[0].set_value(7).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(next(t.value for t in app.dataframe if 'Model 2 pairs' in t.value.columns)['Model 2 pairs'].tolist(), [6]*6)
        app.selectbox[1].set_value(0).run(timeout=60)
        self.assertFalse(app.exception)
        path_data = next(t.value for t in app.dataframe if 'F_FI' in t.value.columns)
        self.assertTrue((path_data['P_link used (full workload)'] == 1).all())
        self.assertTrue((path_data['P_swap'] == 1).all())
        app.selectbox[1].set_value(2).run(timeout=60)
        self.assertFalse(app.exception)
        path_data = next(t.value for t in app.dataframe if 'F_FI' in t.value.columns)
        self.assertEqual(path_data['P_link used (full workload)'].nunique(), 1)
        self.assertTrue((path_data['P_link used (full workload)'] < 1).all())
        app.selectbox[0].set_value('Chain').run(timeout=60)
        self.assertFalse(app.exception)
        figure_titles = [h.value for h in app.subheader if h.value.startswith('Figure ')]
        self.assertEqual([h.split(' · ')[0] for h in figure_titles],
                         ['Figure 5', 'Figure 6', 'Figure 10'])



if __name__=='__main__':
    unittest.main()
