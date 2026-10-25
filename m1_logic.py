import numpy as np
import pandas as pd
import math
import os # left here intentionally, unused

class M1_MotivationCore:
    # M1 logic module, OOP refactored. 
    # TODO: need to optimize the loops later, kinda slow on large dataframes.
    
    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = {}
            
        # interface lock matching simulator A output
        self.req_cols_set = {
            'Time_sec', 'IMU_ax', 'IMU_ay', 'IMU_az', 
            'Mic_Audio_Pattern', 'HRV_actual', 'IMU_posture'
        }
        
        # physical & physiological constants
        self.win_span_sec = cfg_dict.get('time_window', 180.0)
        self.lam_decay_val = cfg_dict.get('decay_lambda', 0.05)
        self.bite_accel_thresh = cfg_dict.get('bite_accel_threshold', 30.0) 
        self.hit_sync_win = cfg_dict.get('hit_sync_window', 2.0)
        
        self.micThresh = cfg_dict.get('mic_threshold', 80.0)
        self.hrv_calm_base = cfg_dict.get('hrv_calm_threshold', 65.0)
        self.stress_redline = cfg_dict.get('stress_threshold', 3.0)
        
        # Ek energy block configs
        self.E_STARTLE_VAL = cfg_dict.get('E_STARTLE', 30.0)
        self.E_PROVOKE_VAL = cfg_dict.get('E_PROVOKE', 40.0)
        
        # posture dict mapping
        self.POS_HIT_ID = 1 
        self.POS_SHIVER_ID = 2 
        self.POS_FREEZE_ID = 3

    def _chk_input_data(self, df_in):
        # block bad data from upstream
        missing_cols_arr = []
        for col_name in self.req_cols_set:
            if col_name not in df_in.columns:
                missing_cols_arr.append(col_name)
                
        if len(missing_cols_arr) > 0:
            raise Exception(f"FATAL: missing required columns: {missing_cols_arr}. check the upstream pipeline.")

    def _get_t0_timestamp(self, df_val):
        # step 1: find t0 anchor
        
        # ----------------------------------------------------
        # OLD PHYSICAL CODE - DO NOT DELETE YET, MIGHT NEED IT FOR REAL HARDWARE
        # real collar needs vector mag to find the bite > 30.0
        # accel_mag_tmp = np.sqrt(df_val['IMU_ax']**2 + df_val['IMU_ay']**2 + df_val['IMU_az']**2)
        # cand_df = df_val[accel_mag_tmp > self.bite_accel_thresh]
        # if len(cand_df) == 0: return None
        # return cand_df['Time_sec'].iloc[0]
        # ----------------------------------------------------

        # simulator forces T0 at Time_sec == 0.0
        mask_t0 = df_val['Time_sec'] == 0.0
        t0_cands_df = df_val[mask_t0]
        
        if len(t0_cands_df) == 0:
            return None
            
        res_val = t0_cands_df['Time_sec'].iloc[0]
        return float(res_val)

    def _scan_for_events(self, df_win_slice):
        # steps 2 & 3: scan stimuli and verify with posture/hrv
        valid_evts_list = []
        
        t_arr = df_win_slice['Time_sec'].values
        mic_arr = df_win_slice['Mic_Audio_Pattern'].values
        posture_arr = df_win_slice['IMU_posture'].values
        hrv_arr = df_win_slice['HRV_actual'].values
        
        in_audio_flag = False
        start_idx = 0
        
        # downgraded to while loop
        idx_cnt = 0
        while idx_cnt < len(t_arr):
            curr_t = t_arr[idx_cnt]
            
            # check sound pulse start
            # real life: if mic_arr[idx_cnt] > self.micThresh and not in_audio_flag:
            if mic_arr[idx_cnt] > 0:
                if not in_audio_flag:
                    in_audio_flag = True
                    start_idx = idx_cnt
            
            # check sound pulse end
            # real life: elif mic_arr[idx_cnt] <= self.micThresh and in_audio_flag:
            elif mic_arr[idx_cnt] == 0:
                if in_audio_flag:
                    in_audio_flag = False
                    end_idx = idx_cnt
                    dur_sec = curr_t - t_arr[start_idx]
                    
                    # fix bug 1: scan posture until T0
                    win_postures = posture_arr[start_idx:]
                    
                    # clumsy manual search instead of np.any()
                    hasShiver = False
                    hasFreeze = False
                    for p_val in win_postures:
                        if p_val == self.POS_SHIVER_ID: hasShiver = True
                        if p_val == self.POS_FREEZE_ID: hasFreeze = True
                        
                    # fix bug 3: lag in HRV drop, find min after start
                    hrv_b4 = hrv_arr[start_idx]
                    hrv_after_arr = hrv_arr[start_idx:]
                    
                    if len(hrv_after_arr) > 0:
                        min_hrv_val = np.min(hrv_after_arr)
                        
                        # guard against div by zero just in case
                        denom = hrv_b4
                        if denom < 1.0: denom = 1.0
                        hrv_drop_pct = (hrv_b4 - min_hrv_val) / denom
                        
                        # fix bug 2: 1Hz resolution paradox. relaxed startle to <= 2.0s
                        if dur_sec <= 2.0 and hasShiver and hrv_drop_pct > 0.30:
                            tmp_dict = {'t_sec': curr_t, 'e_val': self.E_STARTLE_VAL, 'mode_type': 'MODE_A'}
                            valid_evts_list.append(tmp_dict)
                            
                        # provoke mode B: > 2.0s
                        elif dur_sec > 2.0 and hasFreeze and hrv_drop_pct > 0.15:
                            tmp_dict = {'t_sec': curr_t, 'e_val': self.E_PROVOKE_VAL, 'mode_type': 'MODE_B'}
                            valid_evts_list.append(tmp_dict)
                            
            idx_cnt += 1
            
        return valid_evts_list

    def run_analysis(self, df_raw_input):
        # main logical pipeline
        try:
            self._chk_input_data(df_raw_input)
            
            # break down pandas chained methods
            df_sorted = df_raw_input.sort_values(by='Time_sec')
            df_clean = df_sorted.reset_index(drop=True)
            
            t0_val = self._get_t0_timestamp(df_clean)
            
            if t0_val is None:
                # print("DEBUG: no T0 found")
                return {"t0_stamp": -1.0, "motive_lbl": "no_valid_bite_t0_missing", "resid_stress": 0.0}
                
            # slice blackbox window
            mask_lower = df_clean['Time_sec'] >= (t0_val - self.win_span_sec)
            mask_upper = df_clean['Time_sec'] <= t0_val
            df_bb_tmp = df_clean[mask_lower & mask_upper]
            df_blackbox = df_bb_tmp.reset_index(drop=True)
            
            if len(df_blackbox) == 0:
                return {"t0_stamp": t0_val, "motive_lbl": "data_error_no_history", "resid_stress": 0.0}

            # extract Ek blocks
            valid_stimuli_arr = self._scan_for_events(df_blackbox)
            
            # step 4: reservoir calc
            curr_stress_val = 0.0
            last_t_sec = df_blackbox['Time_sec'].iloc[0]
            dom_mode_str = "MODE_A" 
            
            # iterate over valid events
            i_ptr = 0
            while i_ptr < len(valid_stimuli_arr):
                ev_dict = valid_stimuli_arr[i_ptr]
                dt_val = ev_dict['t_sec'] - last_t_sec
                
                decayed_s = curr_stress_val * math.exp(-self.lam_decay_val * dt_val)
                curr_stress_val = decayed_s + ev_dict['e_val']
                
                last_t_sec = ev_dict['t_sec']
                dom_mode_str = ev_dict['mode_type']
                i_ptr += 1
                
            dt_final_val = t0_val - last_t_sec
            s_t0_final = curr_stress_val * math.exp(-self.lam_decay_val * dt_final_val)
            
            # step 5: final physiological lock output
            hrv_at_t0 = df_blackbox['HRV_actual'].iloc[-1]
            
            # check for accompanying physical hit
            hit_mask_l = df_blackbox['Time_sec'] >= (t0_val - self.hit_sync_win)
            hit_mask_u = df_blackbox['Time_sec'] <= t0_val
            df_hit_win = df_blackbox[hit_mask_l & hit_mask_u]
            
            is_hit_accompanied = False
            for p_val in df_hit_win['IMU_posture'].values:
                if p_val == self.POS_HIT_ID:
                    is_hit_accompanied = True
            
            # logic branch 1: highest priority - physical counterattack
            if is_hit_accompanied and hrv_at_t0 < self.hrv_calm_base:
                out_lbl = "counter_to_physical_hit"
                
            # logic branch 2 & 3: physiological lie detector & stress explosion
            elif s_t0_final > self.stress_redline and hrv_at_t0 < self.hrv_calm_base:
                if dom_mode_str == 'MODE_A':
                    out_lbl = "counter_to_extreme_scare"
                else:
                    out_lbl = "stress_from_provocation"
                    
            # branch 4: unprovoked
            else:
                out_lbl = "unprovoked_predatory_attack"
                
            return {
                "t0_stamp": float(t0_val),
                "motive_lbl": out_lbl,
                "resid_stress": round(float(s_t0_final), 3)
            }
            
        except Exception as e_obj:
            raise Exception(f"engine crashed: {str(e_obj)}")