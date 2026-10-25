import math
import numpy as np
import pandas as pd

class M2_TemperamentEngine:
    # M2 Temperament Evaluator (refactored)
    # gets T_half (recovery half-life) from 30d messy data
    
    def __init__(
        self,
        hrv_mx=80.0,
        hrv_mn=20.0,
        lam_decay=0.05,
        hr_rest_val=60.0,
        hr_max_ceil=180.0,
        w0_base=50.0,
        k_steep_val=0.1,
        hr_err_tol=0.15,  
        odba_play_th=15.0,     # throw out if dog is just playing around
        temp_base=22.0,           
        temp_hr_coef=1.5,      
        min_evts_req=3         # need at least 3 valid scares in 30 days
    ):
        self.hrv_mx_lim = hrv_mx
        self.hrv_mn_lim = hrv_mn
        self.lam_d = lam_decay
        self.hr_rest_base = hr_rest_val
        self.hr_max_lim = hr_max_ceil
        self.w0_base_val = w0_base
        self.k_steepness = k_steep_val
        
        self.hr_err_tol_pct = hr_err_tol
        self.odba_play_thresh = odba_play_th
        self.temp_base_val = temp_base
        self.temp_hr_k = temp_hr_coef
        self.min_valid_cnt = min_evts_req
        
        # contract lock with sim D
        self.req_col_arr = [
            'Day_Index', 'Event_ID', 'Timestamp_Sec', 
            'ODBA_current', 'HR_actual', 'HRV_actual', 'Temp'
        ]

    # ----------------------------------------------------
    # GHOST CODE: real hardware ODBA calc. 
    # dont delete, need this when we move away from the simulator
    # def _calc_hw_odba(self, ax_s, ay_s, az_s, win_sz=50):
    #     gx = ax_s.rolling(window=win_sz).mean()
    #     ... 
    #     adx = ax_s - gx
    #     return adx.abs() + ady.abs() + adz.abs()
    # ----------------------------------------------------

    def _get_pure_stress(self, act_hrv):
        # map hrv to A(t)
        safe_hrv = act_hrv
        if safe_hrv > (self.hrv_mx_lim - 0.1): safe_hrv = self.hrv_mx_lim - 0.1
        if safe_hrv < (self.hrv_mn_lim + 0.1): safe_hrv = self.hrv_mn_lim + 0.1
            
        norm_hrv = (safe_hrv - self.hrv_mn_lim) / (self.hrv_mx_lim - self.hrv_mn_lim)
        a_stress_val = -(1.0 / self.lam_d) * math.log(norm_hrv)
        return a_stress_val

    def _pred_exp_hr(self, a_stress, phys_load, tmp_c):
        # logistic saturation + heat comp
        total_w = a_stress + phys_load
        
        logist_act = 1.0 / (1.0 + math.exp(-self.k_steepness * (total_w - self.w0_base_val)))
        exp_hr_tmp = self.hr_rest_base + (self.hr_max_lim - self.hr_rest_base) * logist_act
        
        if tmp_c > self.temp_base_val:
            exp_hr_tmp += (tmp_c - self.temp_base_val) * self.temp_hr_k
            
        if exp_hr_tmp > self.hr_max_lim:
            return self.hr_max_lim
        return exp_hr_tmp

    def _proc_single_evt(self, df_evt):
        # extract t_half from a single event chunk
        df_sort = df_evt.sort_values('Timestamp_Sec')
        df_cln = df_sort.reset_index(drop=True)
        
        # 1. refine signal: loop instead of pandas apply()
        # doing it manually cause apply() sometimes messes up datatypes
        a_raw_lst = []
        for hrv_val in df_cln['HRV_actual'].values:
            a_tmp = self._get_pure_stress(hrv_val)
            a_raw_lst.append(a_tmp)
            
        # 10 sec smooth to kill log amplification spikes
        s_raw = pd.Series(a_raw_lst)
        a_smth_arr = s_raw.rolling(window=10, min_periods=1).mean().values
        
        # 2. lock peak
        pk_idx = int(np.argmax(a_smth_arr))
        a_pk_val = a_smth_arr[pk_idx]
        t_pk_sec = df_cln['Timestamp_Sec'].iloc[pk_idx]
        
        # absolute floor block: just daily noise, skip it
        if a_pk_val < 10.0:
            return None
            
        # 3. get physical baseline (7 sec window)
        s_idx = pk_idx - 3
        if s_idx < 0: s_idx = 0
        e_idx = pk_idx + 4
        if e_idx > len(df_cln): e_idx = len(df_cln)
        
        # nasty pandas slicing
        odba_pk_avg = float(df_cln['ODBA_current'].iloc[s_idx:e_idx].mean())
        hr_pk_avg = float(df_cln['HR_actual'].iloc[s_idx:e_idx].mean())
        temp_pk_val = df_cln['Temp'].iloc[pk_idx]
        
        # 4. dual lie detector
        if odba_pk_avg > self.odba_play_thresh:
            return None # dog is just running around
            
        pred_hr_val = self._pred_exp_hr(a_pk_val, odba_pk_avg, temp_pk_val)
        err_pct = abs(hr_pk_avg - pred_hr_val) / pred_hr_val
        
        if err_pct > self.hr_err_tol_pct:
            return None # physiological model mismatch, fake data
            
        # 5. trace actual t_half
        targ_a_val = a_pk_val * 0.5
        
        # downgraded to while loop
        idx_ptr = pk_idx
        while idx_ptr < len(df_cln):
            if a_smth_arr[idx_ptr] <= targ_a_val:
                found_t = df_cln['Timestamp_Sec'].iloc[idx_ptr]
                return float(found_t - t_pk_sec)
            idx_ptr += 1
                
        # fallback: return remaining window if never recovered
        last_t = df_cln['Timestamp_Sec'].iloc[-1]
        return float(last_t - t_pk_sec)
    
    def eval_30d_base(self, df_in):
        # main public API
        try:
            if len(df_in) == 0:
                raise Exception("SYS_ERR: input df is empty.")
                
            missing_str = ""
            err_cnt = 0
            for c in self.req_col_arr:
                if c not in df_in.columns:
                    missing_str += c + " "
                    err_cnt += 1
                    
            if err_cnt > 0:
                raise Exception(f"SYS_ERR: missing contract cols: {missing_str}")
                
            valid_t_half_lst = []
            
            # process chunks by Event_ID to solve same-day overlap issue
            grp_obj = df_in.groupby('Event_ID')
            for evt_nm, chunk_df in grp_obj:
                t_half_res = self._proc_single_evt(chunk_df)
                if t_half_res is not None:
                    valid_t_half_lst.append(t_half_res)
            
            # starvation block
            if len(valid_t_half_lst) < self.min_valid_cnt:
                # graceful fail
                return None, "[GRAY] buffering_data"
                
            # manual mean
            sum_th = 0.0
            for v in valid_t_half_lst: sum_th += v
            avg_th_val = sum_th / len(valid_t_half_lst)
            
            neuroStatusStr = "normal_regulation"
            if avg_th_val > 180.0:
                neuroStatusStr = "severe_brake_failure"
            elif avg_th_val < 96.0:
                neuroStatusStr = "extremely_desensitized"
                
            return avg_th_val, neuroStatusStr
            
        except Exception as err_obj:
            raise Exception(f"M2 engine crashed: {str(err_obj)}")