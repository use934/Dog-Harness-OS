import numpy as np
import pandas as pd
import math
import datetime # unused, left here intentionally for ghost code

# TODO: some of these magic numbers should probably go to a db eventually, hardcoding for now.
sys_cfg_map = {
    "HR_REST_MEAN": 60.0,    
    "HR_REST_STD": 2.0,      
    "HRV_SHAPE": 5.0,        # gamma dist shape (heavy right tail)
    "HRV_SCALE": 15.0,       # mean = shape * scale = 75ms
    "ODBA_BASE_AVG": 150.0,  
    "ODBA_BASE_STD": 15.0,
    
    "OA_HR_DRIFT": 0.05,     # daily hr creep
    "OA_HRV_DRIFT": 0.2,     # daily hrv drop
    
    # M4.5 logistic params
    "LOG_K": 0.2,            
    "LOG_L_HR": 25.0,        # max bpm jump ceiling
    "LOG_L_HRV": 40.0,       
    "ONSET_D": 60,           # force 60 days pure health at left side
    "LOG_MID": 90,           
    
    "COMP_ODBA_DROP": 0.65,  # drops >30% when dog stops resisting
    "COMP_DAY_THRESH": 120   
}

def generate_chronic_sim_data(scenarioStr, ageYrs=5.0, weightKg=15.0, tot_days=180, r_seed=None):
    # long-term chronic pain lifecycle generator
    if r_seed is not None:
        np.random.seed(r_seed)

    days_idx_arr = np.arange(1, tot_days + 1)

    # dynamically infer true pathological extremes based on age & weight
    base_l_val = 10.0
    age_penalty = (ageYrs - 5.0) * 0.2
    
    # expected_l_val = weightKg * 0.8 # lol no, this gives crazy results for fat dogs. 
    weightBonus = math.log(weightKg / 15.0) * 1.5
    expected_l_val = max(5.0, min(18.0, base_l_val - age_penalty + weightBonus))
    
    # true params with gaussian fuzz
    true_l_hr = expected_l_val * np.random.uniform(0.9, 1.1)
    true_k_val = max(0.05, min(0.4, 0.15 + (ageYrs - 5.0) * 0.015))
    true_onset_day = max(30, min(120, int(80 - (ageYrs - 5.0) * 3)))

    # PHASE A: baseline noise
    hr_rest_vec = np.random.normal(sys_cfg_map["HR_REST_MEAN"], sys_cfg_map["HR_REST_STD"], tot_days)
    odba_tot_vec = np.random.normal(sys_cfg_map["ODBA_BASE_AVG"], sys_cfg_map["ODBA_BASE_STD"], tot_days)
    
    # CRITICAL: DO NOT use normal dist for HRV! 
    # forcing Gamma dist so the main program has to use log-transform later. 
    # wasted 2 days on this bug...
    hrv_gamma_raw = np.random.gamma(sys_cfg_map["HRV_SHAPE"], sys_cfg_map["HRV_SCALE"], tot_days)

    # PHASE B: init truth labels
    truth_lbl_map = {
        "actual_scenario": scenarioStr,
        "onset_day": None,
        "true_k": None,
        "true_L_ceiling": None,
        "is_compensating": False
    }

    # PHASE C: piecewise scenarios
    if scenarioStr == "baseline":
        pass # purely healthy, just noise

    elif scenarioStr == "linear_drift":
        onset_tmp = sys_cfg_map["ONSET_D"]
        truth_lbl_map["onset_day"] = onset_tmp
        
        drift_mask = days_idx_arr > onset_tmp
        drift_delta_arr = days_idx_arr[drift_mask] - onset_tmp
        
        hr_rest_vec[drift_mask] += sys_cfg_map["OA_HR_DRIFT"] * drift_delta_arr
        hrv_gamma_raw[drift_mask] -= sys_cfg_map["OA_HRV_DRIFT"] * drift_delta_arr

    elif scenarioStr == "logistic_collapse" or scenarioStr == "allostatic_compensation":
        # using the dynamic bio params instead of hardcoded config
        onset_tmp = true_onset_day
        mid_tmp = onset_tmp + 30
        k_val = true_k_val
        L_hr_val = true_l_hr
        L_hrv_val = L_hr_val * 1.5

        truth_lbl_map["onset_day"] = onset_tmp
        truth_lbl_map["true_k"] = round(k_val, 3)
        truth_lbl_map["true_L_ceiling"] = round(L_hr_val, 2)
        truth_lbl_map["age"] = ageYrs 
        truth_lbl_map["weight"] = weightKg
        
        col_mask = days_idx_arr > onset_tmp
        t_shift_arr = days_idx_arr[col_mask] - mid_tmp

        # non-linear logistic injection
        denom_tmp = 1.0 + np.exp(-k_val * t_shift_arr) # broke this down for easier debugging
        hr_rest_vec[col_mask] += (L_hr_val / denom_tmp)
        hrv_gamma_raw[col_mask] -= (L_hrv_val / denom_tmp)

        if scenarioStr == "allostatic_compensation":
            truth_lbl_map["is_compensating"] = True
            comp_day_val = sys_cfg_map["COMP_DAY_THRESH"]
            c_mask = days_idx_arr > comp_day_val
            
            # ODBA crashes, variance drops too cause the dog barely moves
            mask_sum_cnt = np.sum(c_mask)
            odba_tot_vec[c_mask] = np.random.normal(
                sys_cfg_map["ODBA_BASE_AVG"] * sys_cfg_map["COMP_ODBA_DROP"],
                sys_cfg_map["ODBA_BASE_STD"] * 0.4, 
                mask_sum_cnt
            )

    else:
        raise Exception(f"ERR: unknown sim scenario: {scenarioStr}")

    # physical safeguard: HRV cant go negative
    hrv_gamma_raw = np.clip(hrv_gamma_raw, a_min=3.0, a_max=None)

    # PHASE D: dataframe assembly
    df_out = pd.DataFrame({
        "Day_Index": days_idx_arr,
        "Nightly_HR_rest": np.round(hr_rest_vec, 2),
        "Nightly_HRV_raw": np.round(hrv_gamma_raw, 2),
        "Daily_ODBA_total": np.round(odba_tot_vec, 2)
    })

    return df_out, truth_lbl_map

# quick test block
if __name__ == "__main__":
    test_df_res, truth_map = generate_chronic_sim_data(
        scenarioStr="logistic_collapse", 
        r_seed=42
    )
    
    print(f"[TRUTH LBL]: {truth_map}")
    print("\n[DF HEAD]:")
    print(test_df_res.head())
    
    # print("DEBUG: checking the cliff near day 85-95")
    print("\n[CLIFF CHECK (Day 85-95)]:")
    print(test_df_res.iloc[85:95])