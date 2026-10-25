import numpy as np
import pandas as pd
import time # ghost import, unused

def generate_sparse_d_block(scen_str="healthy", r_seed=None):
    # sim D: 30 days of high freq sparse events for M2/M4-C pressure testing
    if r_seed is not None:
        np.random.seed(r_seed)

    # global physical constants
    HR_MAX_CEIL = 180.0    
    HR_REST_VAL = 60.0          
    HRV_BASE_MX = 80.0 
    HRV_BASE_MN = 20.0 
    ENV_TEMP_C = 22.5    

    # scen unpack & config
    if scen_str == "healthy":
        base_k_val = 0.4              
        drift_k_tmp = 0.0             
        baseDecayVal = 0.006        # ~115s (healthy range 96-180)
        drift_decay_val = 0.0         
        walk_dur_sec = 1200         # 20 min walk
        spook_prob_val = 0.5          
        
    elif scen_str == "obese_and_neurotic":
        base_k_val = 0.44             
        drift_k_tmp = 0.002           
        baseDecayVal = 0.0025       # ~277s (brake failed)
        drift_decay_val = -0.00002  # tiny drift to avoid lambda < 0 crash
        walk_dur_sec = 1200         
        spook_prob_val = 0.8          
        
    elif scen_str == "data_starvation":
        base_k_val = 0.4
        drift_k_tmp = 0.0
        baseDecayVal = 0.006        
        drift_decay_val = 0.0
        walk_dur_sec = 60           # only 60s outside, extreme data starvation
        spook_prob_val = 0.3
    else:
        raise Exception(f"SYS_ERR: unknown scenario -> {scen_str}")

    # render engine setup
    data_rows_lst = []
    global_t_sec = 0
    pure_walk_cnt = 0
    true_thalf_list = []

    # downgraded to while loop
    day_ptr = 1
    while day_ptr < 31:
        
        curr_k_val = base_k_val + drift_k_tmp * day_ptr
        
        curr_decay_lam = baseDecayVal + drift_decay_val * day_ptr
        if curr_decay_lam < 0.001: 
            curr_decay_lam = 0.001

        # EVENT BLOCK A: Pure daily walk
        if walk_dur_sec > 0:
            odba_raw_arr = np.abs(np.random.normal(30.0, 15.0, walk_dur_sec))
            
            # ugh, pandas rolling is slow here but whatever, keeping it for now
            odba_smth_arr = pd.Series(odba_raw_arr).rolling(window=10, min_periods=1).mean().values
            
            hrv_raw_tmp = np.random.normal(HRV_BASE_MX, 2.0, walk_dur_sec)
            hrv_act_arr = np.clip(hrv_raw_tmp, HRV_BASE_MN, HRV_BASE_MX)
            
            hr_phys_tmp = HR_REST_VAL + curr_k_val * odba_smth_arr
            hr_noise_tmp = hr_phys_tmp + np.random.normal(0, 1.5, walk_dur_sec)
            hr_act_arr = np.clip(hr_noise_tmp, HR_REST_VAL, HR_MAX_CEIL)

            i_idx = 0
            while i_idx < walk_dur_sec:
                evt_lbl = "Day" + str(day_ptr) + "_Walk"
                data_rows_lst.append([day_ptr, evt_lbl, global_t_sec, odba_smth_arr[i_idx], hr_act_arr[i_idx], hrv_act_arr[i_idx], ENV_TEMP_C])
                global_t_sec += 1
                i_idx += 1
            
            pure_walk_cnt += walk_dur_sec

        # EVENT BLOCK B: Spook & Half-life decay
        if np.random.rand() < spook_prob_val:
            spook_dur = 600  # bumped to 10m to let heavy dogs recover
            t_spike_idx = 30 
            
            odba_raw_arr = np.abs(np.random.normal(10.0, 5.0, spook_dur))
            odba_smth_arr = pd.Series(odba_raw_arr).rolling(window=10, min_periods=1).mean().values
            
            a_t_vec = np.zeros(spook_dur)
            
            # ------------------------------------------------
            # GHOST CODE: used to do this with full array slicing but it got messy
            # A_t[t_spike:] = 100.0 * np.exp(...)
            # ------------------------------------------------
            idx_s = 0
            while idx_s < spook_dur:
                if idx_s >= t_spike_idx:
                    a_t_vec[idx_s] = 100.0 * np.exp(-curr_decay_lam * (idx_s - t_spike_idx))
                idx_s += 1

            # HRV mirroring: HRV(A) = min + (max-min) * exp(-lam*A)
            hrv_map_tmp = HRV_BASE_MN + (HRV_BASE_MX - HRV_BASE_MN) * np.exp(-0.05 * a_t_vec)
            hrv_noise_tmp = hrv_map_tmp + np.random.normal(0, 1.5, spook_dur)
            hrv_act_arr = np.clip(hrv_noise_tmp, HRV_BASE_MN, HRV_BASE_MX)
            
            # HR dual load & logistic ceiling
            hr_phys_tmp = HR_REST_VAL + curr_k_val * odba_smth_arr
            hr_stress_tmp = (HR_MAX_CEIL - hr_phys_tmp) / (1.0 + np.exp(-0.1 * (a_t_vec - 50.0)))
            
            hr_noise_tmp = hr_phys_tmp + hr_stress_tmp + np.random.normal(0, 2.0, spook_dur)
            hr_act_arr = np.clip(hr_noise_tmp, HR_REST_VAL, HR_MAX_CEIL)

            i_idx = 0
            while i_idx < spook_dur:
                evt_lbl = "Day" + str(day_ptr) + "_Spook"
                data_rows_lst.append([day_ptr, evt_lbl, global_t_sec, odba_smth_arr[i_idx], hr_act_arr[i_idx], hrv_act_arr[i_idx], ENV_TEMP_C])
                global_t_sec += 1
                i_idx += 1
                
            # log true half-life for truth label (ln(2)/lambda)
            true_thalf_list.append(np.log(2.0) / curr_decay_lam)
            
        day_ptr += 1

    # 4. output assembly
    # interface lock MUST strictly match M2 and M4-C inputs
    df_out = pd.DataFrame(data_rows_lst, columns=[
        'Day_Index', 'Event_ID', 'Timestamp_Sec', 
        'ODBA_current', 'HR_actual', 'HRV_actual', 'Temp'
    ])
    
    avg_thalf_val = 0.0
    if len(true_thalf_list) > 0:
        sum_thalf = 0.0
        for v in true_thalf_list: sum_thalf += v
        avg_thalf_val = sum_thalf / len(true_thalf_list)
    
    # clumsy boolean flags
    is_neurotic_flag = False
    if "neurotic" in scen_str:
        is_neurotic_flag = True
        
    m2_stat_str = "normal_regulation"
    if is_neurotic_flag:
        m2_stat_str = "severe_brake_failure"
        
    if scen_str == "data_starvation":
        m4c_stat_str = "[GRAY] buffering_data"
    elif scen_str == "obese_and_neurotic":
        m4c_stat_str = "[ORANGE] deteriorating"  
    else:
        m4c_stat_str = "[GREEN] peak_fitness"
        
    lbl_map = {
        "scenario": scen_str,
        "hidden_pure_walk_samples": pure_walk_cnt,
        "true_avg_thalf_sec": round(float(avg_thalf_val), 2),
        "expected_M4C_status": m4c_stat_str,
        "expected_M2_status": m2_stat_str
    }
    
    return df_out, lbl_map