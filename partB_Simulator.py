import pandas as pd
import numpy as np
import time # left here intentionally (ghost import)

def generate_metabolic_block(scen_str, r_seed=None):
    # M4.1 cortisol tracking simulator (48 hrs)
    # uncoupled from M4.4. 
    # TODO: need to check if we should add M4.4 coupling back later

    if r_seed is not None:
        np.random.seed(r_seed)
        
    total_m_cnt = 2880  # 48 hrs * 60
    
    # 1. init 48h state canvas
    t_min_arr = np.arange(1, total_m_cnt + 1)
    state_flags = np.ones(total_m_cnt, dtype=int)  # 1 = awake
    
    # hardcode sleep periods 23:00 - 07:00 (state = 0)
    state_flags[0:420] = 0
    state_flags[1380:1860] = 0
    state_flags[2820:total_m_cnt] = 0

    # 2. base physiological normal distributions
    hr_avg_vec = np.zeros(total_m_cnt)
    odba_avg_vec = np.zeros(total_m_cnt)
    a_pulse_arr = np.zeros(total_m_cnt)
    
    mask_sleep = (state_flags == 0)
    mask_awake = (state_flags == 1)
    
    # just filling HR and ODBA to satisfy M4.1 contract interface
    hr_avg_vec[mask_sleep] = np.random.normal(60.0, 2.0, np.sum(mask_sleep))
    odba_avg_vec[mask_sleep] = np.abs(np.random.normal(0.0, 0.5, np.sum(mask_sleep)))
    
    hr_avg_vec[mask_awake] = np.random.normal(75.0, 5.0, np.sum(mask_awake))
    odba_avg_vec[mask_awake] = np.abs(np.random.normal(20.0, 5.0, np.sum(mask_awake)))

    lbl_map_out = {"scenario_type": scen_str}

    # 3. scenario specific overwrites
    if scen_str == "no_stress_base":
        # state = 2 for active to verify k_active constant
        # kinda clunky array concat...
        walk_idx_arr = np.concatenate([
            np.arange(600, 645), np.arange(960, 1005),
            np.arange(2040, 2085), np.arange(2400, 2445)
        ])
        state_flags[walk_idx_arr] = 2
        odba_avg_vec[walk_idx_arr] = np.abs(np.random.normal(150.0, 15.0, len(walk_idx_arr)))
        hr_avg_vec[walk_idx_arr] = np.random.normal(130.0, 8.0, len(walk_idx_arr))
        
        lbl_map_out["expected_m41_alert"] = False

    elif scen_str == "cross_day_stress_residue":
        # base walks
        walk_idx_arr = np.concatenate([
            np.arange(600, 645), np.arange(960, 1005),
            np.arange(2040, 2085), np.arange(2400, 2445)
        ])
        state_flags[walk_idx_arr] = 2
        odba_avg_vec[walk_idx_arr] = np.abs(np.random.normal(150.0, 15.0, len(walk_idx_arr)))
        hr_avg_vec[walk_idx_arr] = np.random.normal(130.0, 8.0, len(walk_idx_arr))
        
        # fragmented sleep simulation (<5 times, <10 mins)
        wake_cnt = np.random.randint(1, 5) 
        
        # downgraded to while loop
        i_tmp = 0
        while i_tmp < wake_cnt:
            w_dur = np.random.randint(2, 10)
            w_start = np.random.randint(1380, 1860 - w_dur)
            
            state_flags[w_start:w_start+w_dur] = 1 
            hr_avg_vec[w_start:w_start+w_dur] = np.random.normal(85.0, 5.0, w_dur)
            i_tmp += 1
            
        # discrete poisson-like high stress pulses (<10)
        pulse_cnt_val = np.random.randint(3, 10) 
        p_records_list = []
        
        # drop pulses on day 1 and night to test cross-day accumulation
        p_times_arr = np.random.choice(np.arange(100, 2000), size=pulse_cnt_val, replace=False)
        
        for pt_idx in p_times_arr:
            intensity_val = np.round(np.random.uniform(80.0, 100.0), 2)
            a_pulse_arr[pt_idx] = intensity_val
            
            tmp_rec = {"t_min": int(pt_idx), "intens_val": float(intensity_val)}
            p_records_list.append(tmp_rec)
            
        # sort by time
        # def sort_func(item): return item['t_min']
        # p_records_list.sort(key=sort_func)
        p_records_list = sorted(p_records_list, key=lambda x_obj: x_obj['t_min'])
            
        lbl_map_out["expected_m41_alert"] = True
        lbl_map_out["injected_pulses_arr"] = p_records_list

    else:
        raise Exception("ERR: unknown scenario. use 'no_stress_base' or 'cross_day_stress_residue'.")

    # 4. output data assembly
    # NOTE: keeping exact DataFrame col names to match downstream M4.1 requirements
    df_out = pd.DataFrame({
        "Timestamp_Min": t_min_arr,
        "State": state_flags,
        "Avg_ODBA": odba_avg_vec,
        "Avg_HR": hr_avg_vec,
        "A_Pulse": a_pulse_arr
    })
    
    return df_out, lbl_map_out