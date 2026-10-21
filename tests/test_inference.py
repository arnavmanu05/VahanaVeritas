import unittest
import numpy as np

from models.market_regressor import MarketValuationEngine, get_launch_msrp, get_brand_tier
from models.mechanical_lstm import MechanicalHealthEngine, generate_telemetry_preset
from models.multimodal_nlp import MultimodalConditionEngine
from models.fusion_engine import calculate_late_fusion, get_lemon_risk_assessment, ARCHETYPES
from utils.report_generator import generate_inspection_certificate

class TestVahanaVeritasPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.market_engine = MarketValuationEngine()
        cls.lstm_engine = MechanicalHealthEngine()
        cls.nlp_engine = MultimodalConditionEngine()

    def test_market_regressor_predictions(self):
        specs = {
            "brand": "Hyundai",
            "model": "Creta",
            "model_year": 2020,
            "km_driven": 45000,
            "fuel_type": "Diesel",
            "transmission": "Manual",
            "engine_cc": 1497.0,
            "max_power_bhp": 113.0
        }
        res = self.market_engine.predict(specs)
        self.assertIn('depreciation_ratio', res)
        self.assertIn('baseline_price_inr', res)
        self.assertGreater(res['baseline_price_inr'], 100000.0)
        self.assertLess(res['baseline_price_inr'], res['launch_msrp'] * 1.05)
        self.assertTrue(0.10 <= res['depreciation_ratio'] <= 1.05)

    def test_mechanical_health_lstm_bounds(self):
        pristine_seq = generate_telemetry_preset('pristine')
        self.assertEqual(pristine_seq.shape, (30, 14))

        res_pristine = self.lstm_engine.predict(pristine_seq)
        mhi_p = res_pristine['mechanical_health_index']
        self.assertTrue(0.0 <= mhi_p <= 1.0)

        lemon_seq = generate_telemetry_preset('lemon')
        res_lemon = self.lstm_engine.predict(lemon_seq)
        mhi_l = res_lemon['mechanical_health_index']
        self.assertTrue(0.0 <= mhi_l <= 1.0)

        # Pristine sequence must exhibit higher MHI than degraded sequence
        self.assertGreater(mhi_p, mhi_l)

    def test_multimodal_nlp_condition_score(self):
        good_text = "Regular periodic service completed. All fluids fresh, brake pads nominal, passed 200-point inspection."
        good_tab = {"tire_tread_mm": 7.5, "brake_pad_wear_pct": 15.0, "fluid_condition": 1.0, "service_adherence_ratio": 0.98}
        res_good = self.nlp_engine.predict(good_text, good_tab)
        cs_good = res_good['condition_score']
        self.assertTrue(0.0 <= cs_good <= 1.0)

        bad_text = "Engine overheating, blown head gasket detected, cylinder head cracked, severe oil sludge."
        bad_tab = {"tire_tread_mm": 1.5, "brake_pad_wear_pct": 95.0, "fluid_condition": 0.05, "service_adherence_ratio": 0.20}
        res_bad = self.nlp_engine.predict(bad_text, bad_tab)
        cs_bad = res_bad['condition_score']
        self.assertTrue(0.0 <= cs_bad <= 1.0)

        self.assertGreater(cs_good, cs_bad)

    def test_late_fusion_exact_math(self):
        base_price = 800000.0
        w1, w2 = 0.15, 0.15

        # 1. Neutral Baseline Case (MHI=0.50, CS=0.50)
        res_neutral = calculate_late_fusion(base_price, 0.50, 0.50, w1, w2)
        self.assertAlmostEqual(res_neutral['multiplier'], 1.000, places=3)
        self.assertAlmostEqual(res_neutral['final_price'], 800000.0, places=1)
        self.assertAlmostEqual(res_neutral['delta_inr'], 0.0, places=1)
        self.assertAlmostEqual(res_neutral['delta_pct'], 0.0, places=2)

        # 2. Pristine Case (MHI=0.95, CS=0.96)
        # Multiplier = 1.0 + 0.15*(0.45) + 0.15*(0.46) = 1.0 + 0.0675 + 0.0690 = 1.1365
        res_pristine = calculate_late_fusion(base_price, 0.95, 0.96, w1, w2)
        expected_mult = 1.0 + (0.15 * 0.45) + (0.15 * 0.46) # 1.1365
        self.assertAlmostEqual(res_pristine['multiplier'], expected_mult, places=4)
        self.assertAlmostEqual(res_pristine['final_price'], 800000.0 * expected_mult, places=1)
        self.assertAlmostEqual(res_pristine['delta_inr'], 800000.0 * (expected_mult - 1.0), places=1)
        self.assertAlmostEqual(res_pristine['delta_pct'], (expected_mult - 1.0) * 100.0, places=2)

        # 3. Lemon Case (MHI=0.12, CS=0.15)
        # Multiplier = 1.0 + 0.15*(-0.38) + 0.15*(-0.35) = 1.0 - 0.0570 - 0.0525 = 0.8905
        res_lemon = calculate_late_fusion(base_price, 0.12, 0.15, w1, w2)
        expected_lemon_mult = 1.0 + (0.15 * (0.12 - 0.50)) + (0.15 * (0.15 - 0.50)) # 0.8905
        self.assertAlmostEqual(res_lemon['multiplier'], expected_lemon_mult, places=4)
        self.assertAlmostEqual(res_lemon['final_price'], 800000.0 * expected_lemon_mult, places=1)
        self.assertAlmostEqual(res_lemon['delta_inr'], 800000.0 * (expected_lemon_mult - 1.0), places=1)
        self.assertAlmostEqual(res_lemon['delta_pct'], (expected_lemon_mult - 1.0) * 100.0, places=2)

        # Verify Lemon Gap resolution: difference between pristine and lemon is ~24.6%
        spread = res_pristine['multiplier'] - res_lemon['multiplier']
        self.assertAlmostEqual(spread, 0.246, places=3)

    def test_end_to_end_certificate_generation(self):
        specs = {
            "brand": "Honda",
            "model": "City",
            "model_year": 2021,
            "km_driven": 25000,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "engine_cc": 1498.0,
            "max_power_bhp": 119.0
        }
        market_res = self.market_engine.predict(specs)
        seq = generate_telemetry_preset('pristine')
        mhi_res = self.lstm_engine.predict(seq)

        log_text = "Scheduled 20k-km service completed on time. All inspection points nominal."
        tab_inputs = {"tire_tread_mm": 7.0, "brake_pad_wear_pct": 18.0, "fluid_condition": 1.0, "service_adherence_ratio": 1.0}
        nlp_res = self.nlp_engine.predict(log_text, tab_inputs)

        fusion_res = calculate_late_fusion(
            baseline_price=market_res['baseline_price_inr'],
            mhi=mhi_res['mechanical_health_index'],
            cs=nlp_res['condition_score']
        )
        fusion_res['launch_msrp'] = market_res['launch_msrp']

        cert = generate_inspection_certificate(specs, fusion_res, nlp_res, mhi_res)
        self.assertIn("VAHANAVERITAS AUTOMOTIVE VALUATION CERTIFICATE", cert)
        self.assertIn("Honda City", cert)
        self.assertIn("CERTIFIED CONDITION-ADJUSTED VALUE", cert)

if __name__ == '__main__':
    unittest.main()
