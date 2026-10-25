import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import math # left intentionally inside methods in original, moved up here for cleaner look but kept some below

class M45_PredictiveEngine:
    # M4.5 Chronic Predictor (OOP version)
    # combines ALI dimensionality reduction, gated state machine, 
    # and left-anchored Logistic prediction cone.
    
    @staticmethod
    def _calc_priors(age_val, weight_val):
        # calculate physio ceiling L priors based on allometric scaling
        import math # redundant import, typical dirty code habit
        
        base_l_bpm = 10.0
        age_pen = (age_val - 5.0) * 0.2
        weight_bon = math.log(weight_val / 15.0) * 1.5
        
        # 1. calc physical bpm jump limit
        exp_l_bpm = base_l_bpm - age_pen + weight_bon
        
        if exp_l_bpm < 5.0: exp_l_bpm = 5.0
        if exp_l_bpm > 18.0: exp_l_bpm = 18.0
        
        # 2. dimension transform to ALI space
        # assuming std_hr = 2.0, HRV decay adds ~20% extra to z-score
        exp_ali_l = (exp_l_bpm / 2.0) * 1.2
        
        # return (lower bound jack, expected, upper bound ceiling)
        return exp_ali_l * 0.75, exp_ali_l, exp_ali_l * 1.25

    def __init__(self, cfg_map=None):
        if cfg_map is None:
            cfg_map = {}
            
        # pop/individual base constants for ALI
        self.mu_hr_val = cfg_map.get("mu_hr", 60.0)
        self.std_hr_val = cfg_map.get("std_hr", 5.0)
        self.mu_hrv_ln = cfg_map.get("mu_hrv_ln", 4.2)  # ~ln(65)
        self.std_hrv_ln = cfg_map.get("std_hrv_ln", 0.3)
        
        # Gate 0 absolute health locks
        self.health_hr_max = cfg_map.get("healthy_hr_max", 65.0)
        self.health_hrv_min = cfg_map.get("healthy_hrv_min", 60.0)
        
        # gate routing & slope thresh
        self.x_extreme_val = cfg_map.get("x_extreme", 3.0)
        self.sig_speed = cfg_map.get("sigma_speed", 0.05)
        
        self.speed_th = 3 * self.sig_speed
        if self.speed_th < 0.15: self.speed_th = 0.15
            
        self.health_ali_base = cfg_map.get("health_baseline_ali", 1.5)
        self.k_min_slope = cfg_map.get("k_min", 0.05)
        
        # prediction anchor & bounds
        self.anchor_d_cnt = cfg_map.get("anchor_days", 60) # left weight to prevent straight line divergence
        
        # internal FSM states
        self.t0_idx_val = None
        self.curr_fsm_state = "GATE_0_SLEEP"
        self.req_col_arr = ['Day_Index', 'Nightly_HR_rest', 'Nightly_HRV_raw']

    def _reset_fsm(self):
        self.t0_idx_val = None
        self.curr_fsm_state = "GATE_0_SLEEP"

    def _chk_iface(self, df_in):
        missing_arr = []
        for c in self.req_col_arr:
            if c not in df_in.columns:
                missing_arr.append(c)
                
        if len(missing_arr) > 0:
            raise Exception(f"SYS_BLOCK: missing mandatory cols: {missing_arr}.")

    def _get_ali_array(self, hr_arr, hrv_arr):
        # Layer 1: Data refining
        # BIG FIX: don't let log(0) blow up the whole pipeline
        hrv_safe_arr = np.clip(hrv_arr, a_min=1.0, a_max=None)
        hrv_ln_arr = np.log(hrv_safe_arr)
        
        z_hr_arr = (hr_arr - self.mu_hr_val) / self.std_hr_val
        z_hrv_arr = (hrv_ln_arr - self.mu_hrv_ln) / self.std_hrv_ln
        
        # scissors diff
        res_ali = z_hr_arr - z_hrv_arr
        return res_ali

    @staticmethod
    def _logistic_func(x_arr, L_val, k_val, x0_val):
        # np.clip prevents exp overflow crash
        clipped_power = np.clip(-k_val * (x_arr - x0_val), -100, 100)
        e_term = np.exp(clipped_power)
        return L_val / (1.0 + e_term)

    def _run_curve_fit(self, x_vec, y_vec, p_l_min, p_l_max):
        # Layer 3: Constrained Non-linear Fit Engine (Bio-prior version)
        
        hist_max_val = np.nanmax(y_vec)
        dyn_l_max = p_l_max
        if (hist_max_val * 1.2) > dyn_l_max:
            dyn_l_max = hist_max_val * 1.2
            
        dyn_l_min = p_l_min
        if (dyn_l_max - 0.1) < dyn_l_min:
            dyn_l_min = dyn_l_max - 0.1

        l_init = np.clip(hist_max_val, dyn_l_min, dyn_l_max)
        p0_guess = [l_init, 0.5, np.median(x_vec)]
        
        # lock optimizer inside bio bounds
        bound_low = [dyn_l_min, 0.0, -np.inf]
        bound_high = [dyn_l_max, np.inf, np.inf]
        
        try:
            # keeping huber loss to filter single day noise spikes
            popt_res, pcov_res = curve_fit(
                self._logistic_func, x_vec, y_vec, p0=p0_guess,
                bounds=(bound_low, bound_high), method='trf', loss='huber'
            )
            p_err = np.sqrt(np.diag(pcov_res))
            
            b_L = popt_res[0]
            b_k = popt_res[1]
            b_x0 = popt_res[2]
            
            # physical error shackles to prevent early astronomical variance
            # 1. slope err limit
            err_k = p_err[1]
            if err_k > (b_k * 0.5 + 0.1): err_k = b_k * 0.5 + 0.1
                
            # 2. time err limit (15 days max)
            err_x0 = p_err[2]
            if err_x0 > 15.0: err_x0 = 15.0
                
            # 3. ceiling err limit
            err_L = p_err[0]
            if err_L > (b_L * 0.1): err_L = b_L * 0.1
                
            b_k_best = b_k - err_k
            if b_k_best < 0.01: b_k_best = 0.01
            
            b_l_best = b_L - err_L
            if b_l_best < 1.0: b_l_best = 1.0
                
            return {
                "fit_ok": True,
                "p_dict": {"L": b_L, "k": b_k, "x0": b_x0},
                "worst_c": {"L": b_L + err_L, "k": b_k + err_k, "x0": b_x0 - err_x0},
                "best_c": {"L": b_l_best, "k": b_k_best, "x0": b_x0 + err_x0}
            }
        except Exception as e_obj:
            # print(f"DEBUG: fit diverged: {e_obj}")
            return {"fit_ok": False, "err_msg": "Matrix diverged."}

    def process_time_window(self, df_slice, age_y=5.0, weight_k=15.0):
        # core external API
        # strict rule: no look ahead. play the FSM from the start.
        self._chk_iface(df_slice)
        
        d_idx_arr = df_slice['Day_Index'].values
        hr_raw_arr = df_slice['Nightly_HR_rest'].values
        hrv_raw_arr = df_slice['Nightly_HRV_raw'].values
        
        ali_raw = self._get_ali_array(hr_raw_arr, hrv_raw_arr)
        total_n = len(ali_raw)
        
        # fix 3: use EMA to absorb day noise but keep path inertia
        # breaking down pandas chains
        s_hr = pd.Series(hr_raw_arr)
        hr_ema_arr = s_hr.ewm(span=3, adjust=False).mean().values
        
        s_hrv = pd.Series(hrv_raw_arr)
        hrv_ema_arr = s_hrv.ewm(span=3, adjust=False).mean().values
        
        # CRITICAL: EMA on ALI directly for FSM speed check
        s_ali = pd.Series(ali_raw)
        ali_ema_arr = s_ali.ewm(span=3, adjust=False).mean().values
        
        trig_l3_flag = False
        curr_alarm_str = "NONE"
        
        end_idx_val = total_n  
        self._reset_fsm()
        
        # Layer 2: Dynamic Gated FSM
        idx = 1
        while idx < total_n:
            if np.isnan(ali_raw[idx]) or np.isnan(ali_raw[idx-1]):
                idx += 1
                continue 
                
            # fix 2: back to first principles for Gate 0
            if ali_ema_arr[idx] < 0.5:
                self._reset_fsm()
                idx += 1
                continue
                
            if self.curr_fsm_state == "GATE_0_SLEEP":
                delta_ali = ali_ema_arr[idx] - ali_ema_arr[idx-1]
                if delta_ali >= self.x_extreme_val:
                    curr_alarm_str = "FATAL_ACUTE"
                    self.t0_idx_val = idx 
                    self.curr_fsm_state = "GATE_2_WAIT_T3"
                elif delta_ali > 0.05: 
                    self.t0_idx_val = idx
                    self.curr_fsm_state = "GATE_2_WAIT_T3"
                    
            elif self.curr_fsm_state == "GATE_2_WAIT_T3":
                days_diff = d_idx_arr[idx] - d_idx_arr[self.t0_idx_val]
                if days_diff >= 3:
                    delta_3d = ali_ema_arr[idx] - ali_ema_arr[self.t0_idx_val]
                    if delta_3d <= 0:
                        self._reset_fsm()
                    elif delta_3d > 1.0:
                        curr_alarm_str = "MODERATE_ACUTE"
                        self.curr_fsm_state = "GATE_3_SLIDING_WINDOW"
                    else:
                        self.curr_fsm_state = "GATE_3_SLIDING_WINDOW"
                        
            elif self.curr_fsm_state == "GATE_3_SLIDING_WINDOW":
                days_diff = d_idx_arr[idx] - d_idx_arr[self.t0_idx_val]
                
                # fix 4: 14-day self destruct removed. 10 days is enough to roll.
                if days_diff >= 10:
                    win_mask = ~np.isnan(ali_ema_arr[self.t0_idx_val:idx+1])
                    v_cnt = np.sum(win_mask)
                    
                    if v_cnt >= 7:
                        x_val_arr = d_idx_arr[self.t0_idx_val:idx+1][win_mask] - d_idx_arr[self.t0_idx_val]
                        y_val_arr = ali_ema_arr[self.t0_idx_val:idx+1][win_mask]
                        
                        slope_val, _, _, _, _ = linregress(x_val_arr, y_val_arr)
                        m_y = np.mean(y_val_arr)
                        
                        last_3 = y_val_arr[-3:]
                        is_cliff = (last_3[0] - last_3[-1]) > 0.5 
                        
                        # fix 5: lower early height req. as long as trend is pos, pop L3.
                        if m_y > 0.5 and slope_val > self.k_min_slope and not is_cliff:
                            trig_l3_flag = True
                            end_idx_val = idx + 1 
                            break
            idx += 1

        # Layer 3: S-curve anchor & cone gen
        out_map = {
            "alarm_state": curr_alarm_str,
            "trigger_l3": trig_l3_flag,
            "pred_cone_data": None
        }

        if trig_l3_flag and self.t0_idx_val is not None:
            s_idx = self.t0_idx_val - self.anchor_d_cnt
            if s_idx < 0: s_idx = 0
            
            x_ser = d_idx_arr[s_idx:end_idx_val]
            y_ser = ali_raw[s_idx:end_idx_val]
            
            clean_mask = ~np.isnan(y_ser)
            x_cln = x_ser[clean_mask]
            y_cln = y_ser[clean_mask]
            
            if len(x_cln) >= 10: 
                x_shifted = x_cln - x_cln[0] 
                
                # 1. bio jacks & ceilings
                l_m_p, _, l_mx_p = self._calc_priors(age_y, weight_k)
                
                # 2. inject bounds
                f_res = self._run_curve_fit(x_shifted, y_cln, l_m_p, l_mx_p)
                
                if f_res.get("fit_ok"):
                    # fix 2: reverse shift back to physical days
                    base_x0_val = x_cln[0]
                    f_res["p_dict"]["x0"] += base_x0_val
                    f_res["worst_c"]["x0"] += base_x0_val
                    f_res["best_c"]["x0"] += base_x0_val
                    
                    out_map["pred_cone_data"] = f_res
                else:
                    # diverged, strip hallucination
                    out_map["trigger_l3"] = False
                    out_map["alarm_state"] = "OBSERVATION_DUE_TO_DIVERGENCE"

        return out_map
        
    @staticmethod
    def calc_annual_base(v_rest_days, theor_dep_val, days_worn_cnt, prev_year_h0):
        # Layer 5: new absolute pure baseline H0
        if len(v_rest_days) == 0:
            return prev_year_h0, False
            
        act_limit = np.percentile(v_rest_days, 90)
        lam_w = 0.05 if days_worn_cnt > 300 else 0.5 
        new_h0_val = lam_w * theor_dep_val + (1 - lam_w) * act_limit
        
        # rigid physical lock: block >15% aging (subclinical)
        if new_h0_val < (prev_year_h0 * 0.85):
            return prev_year_h0 * 0.95, True 
            
        return new_h0_val, False