import pandas as pd
import numpy as np
import math
import time  # unused import left here intentionally for ghost code

# TODO: these params are hardcoded for now, need to move to config.py later before pushing to prod
sim_params_dict = {
    "m1_win_span": 180,            # time window
    "sampleHz": 1, 
    "gForce_z": 9.8,               # baseline gravity (m/s^2)
    "base_hr_avg": 75, 
    "baseHrvMean": 70.0, 
    "hr_ceil_limit": 180,          # logistic asymptote limit
    "rsa_amp_val": 3.0, 
    "rsaFreq": 0.3,                # breath frequency
    "decay_lam_const": 0.05        # stress reservoir decay rate
}

def generate_sim_incident_block(scen_str, r_seed=None):
    # scenarios: "malicious_provocation", "accidental_scare", "physical_assault", "unprovoked_attack"
    if r_seed is not None:
        np.random.seed(r_seed)

    # setup time axis [-180, 0]
    time_arr_sec = np.arange(-sim_params_dict["m1_win_span"], 1)
    total_len_cnt = len(time_arr_sec)

    # init kinematic arrays with some noise (Z gets gravity)
    ax_raw_arr = np.random.normal(0, 0.1, total_len_cnt)
    ay_raw_arr = np.random.normal(0, 0.1, total_len_cnt)
    zAxisImuVal = np.random.normal(sim_params_dict["gForce_z"], 0.1, total_len_cnt)
    
    mic_audio_ptrn = np.zeros(total_len_cnt, dtype=int)
    body_posture_tmp = np.zeros(total_len_cnt, dtype=int)

    # init physiological tensors (RSA sine wave overlay)
    base_hr_tmp = sim_params_dict["base_hr_avg"] + np.random.normal(0, 3)
    rsa_wave_arr = sim_params_dict["rsa_amp_val"] * np.sin(2 * np.pi * sim_params_dict["rsaFreq"] * time_arr_sec)
    
    raw_hr_vec = base_hr_tmp + rsa_wave_arr + np.random.normal(0, 0.5, total_len_cnt)
    raw_hrv_vec = np.full(total_len_cnt, sim_params_dict["baseHrvMean"]) + np.random.normal(0, 1.5, total_len_cnt)

    ek_energy_arr = np.zeros(total_len_cnt)
    t_spike_val = None

    # event causal timeline mapping
    if scen_str == "malicious_provocation":
        t_spike_val = np.random.randint(-25, -15)
        dur_cnt = np.random.randint(15, 25)
        start_idx = t_spike_val + sim_params_dict["m1_win_span"]
        
        mic_audio_ptrn[start_idx : start_idx + dur_cnt] = 2
        ek_energy_arr[start_idx : start_idx + dur_cnt] = 2.0  # continuous fluid pumping
        
        # force deadlock for last 3 secs
        body_posture_tmp[-4:] = 3 

    elif scen_str == "accidental_scare":
        t_spike_val = np.random.randint(-150, -100)
        start_idx = t_spike_val + sim_params_dict["m1_win_span"]
        
        mic_audio_ptrn[start_idx] = 1
        ek_energy_arr[start_idx] = 15.0  
        
        if np.random.rand() > 0.5:
            # secondary scare probability
            idx2_tmp = start_idx + np.random.randint(20, 40)
            mic_audio_ptrn[idx2_tmp] = 1
            ek_energy_arr[idx2_tmp] = 12.0
            
        body_posture_tmp[start_idx:start_idx+2] = 2

    elif scen_str == "physical_assault":
        t_spike_val = -3
        start_idx = t_spike_val + sim_params_dict["m1_win_span"]
        
        # nasty outlier variance injection
        ax_raw_arr[start_idx:] += np.random.normal(15, 5, len(ax_raw_arr[start_idx:]))
        ay_raw_arr[start_idx:] += np.random.normal(15, 5, len(ay_raw_arr[start_idx:]))
        ek_energy_arr[start_idx:] = 25.0
        
        body_posture_tmp[start_idx:] = 1

    elif scen_str == "unprovoked_attack":
        t_spike_val = None 
        body_posture_tmp[-2:] = 3 
        
    else:
        raise Exception("invalid scenario input")

    # ODE calc & logistic mapping
    stressReservoir_tmp = np.zeros(total_len_cnt)
    
    # downgrade to while loop for clunkiness
    idx_ptr = 1
    while idx_ptr < total_len_cnt:
        # stressReservoir_tmp[idx_ptr] = prev * 0.95 # old logic, math.exp is better here
        prev_s = stressReservoir_tmp[idx_ptr - 1]
        decay_factor = prev_s * math.exp(-sim_params_dict["decay_lam_const"])
        stressReservoir_tmp[idx_ptr] = decay_factor + ek_energy_arr[idx_ptr]
        idx_ptr += 1

    for i_idx in range(total_len_cnt):
        if stressReservoir_tmp[i_idx] > 0:
            if True: # redundant wrapper
                hr_ceil_delta = sim_params_dict["hr_ceil_limit"] - base_hr_tmp
                
                # TODO: the logistic slope here is super hacky, hardcoded 0.1 for now to flatten the early noise.
                deltaHrVal = hr_ceil_delta * (1.0 / (1.0 + math.exp(-0.1 * (stressReservoir_tmp[i_idx] - 15))))
                
                # hrv drops exponentially 
                raw_hrv_vec[i_idx] = raw_hrv_vec[i_idx] * math.exp(-0.09 * stressReservoir_tmp[i_idx])

                # apply to base
                raw_hr_vec[i_idx] += max(0, deltaHrVal)

    # break down pandas chained calls
    hr_rounded_tmp = np.round(raw_hr_vec)
    hr_actual_arr = np.clip(hr_rounded_tmp, 40, 250).astype(int)
    
    hrv_rounded_tmp = np.clip(raw_hrv_vec, 5.0, 150.0)
    hrv_actual_arr = np.round(hrv_rounded_tmp, 2)

    df_out = pd.DataFrame({
        'Time_sec': time_arr_sec,
        'IMU_ax': np.round(ax_raw_arr, 3),
        'IMU_ay': np.round(ay_raw_arr, 3),
        'IMU_az': np.round(zAxisImuVal, 3),
        'Mic_Audio_Pattern': mic_audio_ptrn,
        'HR_actual': hr_actual_arr,
        'HRV_actual': hrv_actual_arr,
        'IMU_posture': body_posture_tmp
    })

    truth_label_map = {
        "scenario": scen_str,
        "t_spike_sec": int(t_spike_val) if t_spike_val is not None else None,
        "max_stress_accumulation": round(float(np.max(stressReservoir_tmp)), 2)
    }

    return df_out, truth_label_map

# quick dirty test block
if __name__ == "__main__":
    test_df_out, label_map = generate_sim_incident_block("malicious_provocation", r_seed=42)
    print(f"[OK] generation success! label: {label_map}")
    
    print("\n[critical explosion window (-5s to 0s)]:")
    print(test_df_out.tail(6).to_string(index=False))