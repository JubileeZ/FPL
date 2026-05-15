import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fpl_engine.optimization import (
    generate_all_duels_matrix,
    calculate_overall_score,
    minutes_composite_loss
)
from fpl_engine.scenarios import generate_scenario_tensor

def run_tests():
    log_lines = []
    log_lines.append("=== FPL ENGINE STATISTICAL TEST SUITE ===")
    
    # Test 1
    try:
        df = pd.DataFrame({
            'id_player': [1, 2],
            'web_name': ['Star', 'Bench'],
            'Perf_IDX': [15.0, 1.0],
            'score_std': [0.1, 0.1]
        })
        duels = generate_all_duels_matrix(df)
        assert len(duels) == 2
        star_vs_bench = duels[(duels['id_player_A'] == 1) & (duels['id_player_B'] == 2)].iloc[0]
        assert star_vs_bench['Win_%'] > 99.0
        assert not np.isnan(star_vs_bench['Win_%'])
        log_lines.append("[PASS] test_skellam_parameterization_hybrid_fallback")
    except Exception as e:
        log_lines.append(f"[FAIL] test_skellam_parameterization_hybrid_fallback: {e}")

    # Test 2
    try:
        empty_df = pd.DataFrame(columns=['target', 'pred', 'gameweek'])
        score = calculate_overall_score(empty_df, 'target', 'pred')
        assert isinstance(score, float)
        assert score == 0.0
        log_lines.append("[PASS] test_calculate_overall_score_return_type")
    except Exception as e:
        log_lines.append(f"[FAIL] test_calculate_overall_score_return_type: {e}")

    # Test 3
    try:
        np.random.seed(42)
        df = pd.DataFrame({
            'id_player': [1, 1, 2, 2, 3, 3],
            'gameweek': [1, 2, 1, 2, 1, 2],
            'Perf_IDX': [5.0, 6.0, 3.0, 4.0, 4.5, 5.5],
            'score_std': [2.0, 2.5, 1.5, 2.0, 1.8, 2.2]
        })
        corr_matrix = np.eye(6)
        component_corr = {'corr_matrix': corr_matrix}
        tensor = generate_scenario_tensor(df, component_corr, n_scenarios=1000, seed=42)
        assert tensor.shape == (3, 2, 1000)
        
        p1_gw1 = tensor[0, 0, :]
        p2_gw1 = tensor[1, 0, :]
        correlation = np.corrcoef(p1_gw1, p2_gw1)[0, 1]
        assert abs(correlation) < 0.1
        log_lines.append(f"[PASS] test_scenario_tensor_independence (corr={correlation:.4f})")
    except Exception as e:
        log_lines.append(f"[FAIL] test_scenario_tensor_independence: {e}")

    # Test 4
    try:
        y_true = np.array([0, 0])
        y_pred_high = np.array([90, 90])
        y_pred_capped = np.array([30, 30])
        
        loss_high = minutes_composite_loss(y_true, y_pred_high)
        loss_capped = minutes_composite_loss(y_true, y_pred_capped)
        
        assert loss_capped < loss_high
        assert loss_high > 0
        log_lines.append("[PASS] test_minutes_composite_loss_rest_penalty")
    except Exception as e:
        log_lines.append(f"[FAIL] test_minutes_composite_loss_rest_penalty: {e}")

    log_content = "\n".join(log_lines)
    print(log_content)
    
    with open('scratch/test_logs.txt', 'w') as f:
        f.write(log_content)

if __name__ == '__main__':
    run_tests()
