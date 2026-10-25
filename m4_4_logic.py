import numpy as np
import pandas as pd
import math

class M4C_CardioRadar:
    # M4-C Cardio & Obesity Radar
    # extracts pure kinematic work and fits HR response slope K
    
    def __init__(self,
                 base_k_val=0.4,
                 min_req_samples=120,
                 max_a_stress=20.0,
                 stress_win_s=300,
                 min_odba_val=3.0,
                 hrv_base_mx=80.0,
                 hrv_base_mn=20.0,
                 decay_lam=0.05):
        
        self.baselineK = base_k_val
        self.min_samp_cnt = min_req_samples
        self.max_stress_thresh = max_a_stress
        self.stress_win_sec = stress_win_s
        self.min_phys_odba = min_odba_val
        self.hrv_mx_lim = hrv_base_mx
        self.hrv_mn_lim = hrv_base_mn
        self.lam_decay = decay_lam

        # contract lock with Simulator D
        self.req_col_arr = [
            'Day_Index', 'Event_ID', 'Timestamp_Sec', 
            'ODBA_current', 'HR_actual', 'HRV_actual', 'Temp'
        ]

    # -------------------------------------------------------------------------
    # OLD C++ PORT: raw ODBA extraction logic. 
    # we get ODBA directly from firmware/simulator now, so this is dead code.
    # keeping it here just in case we need to re-implement edge calculation.
    #
    # def _calc_raw_odba(self, xyz_tuple_list):
    #     if not xyz_tuple_list: return 0.0
    #     sz = len(xyz_tuple_list)
    #     sx = 0.0; sy = 0.0; sz_a = 0.0
    #     for ax, ay, az in xyz_tuple_list:
    #         sx += ax; sy += ay; sz_a += az
    #     gx = sx / sz; gy = sy / sz; gz = sz_a / sz
    #     last_x, last_y, last_z = xyz_tuple_list[-1]
    #     return abs(last_x - gx) + abs(last_y - gy) + abs(last_z - gz)
    # -------------------------------------------------------------------------

    def _chk_contract(self, df_in):
        # dumpy interface defense
        missing_str = ""
        err_c = 0
        for col_name in self.req_col_arr:
            if col_name not in df_in.columns:
                missing_str += col_name + " "
                err_c += 1
                
        if err_c > 0:
            raise Exception(f"SYS_ERR: missing strict cols: {missing_str}")
            
        if len(df_in) == 0:
            raise Exception("SYS_ERR: df is empty, radar aborted.")

    def _get_psy_stress(self, hrv_ser):
        # reverse bio-math: A(t) from HRV
        eps_val = 1e-5
        low_bnd = self.hrv_mn_lim + eps_val
        
        hrv_clp = hrv_ser.clip(lower=low_bnd, upper=self.hrv_mx_lim)
        ratio_ser = (hrv_clp - self.hrv_mn_lim) / (self.hrv_mx_lim - self.hrv_mn_lim)
        
        a_t_ser = -(1.0 / self.lam_decay) * np.log(ratio_ser)
        return a_t_ser.fillna(0.0)

    def extract_pure_blocks(self, df_raw):
        # Stage 1: emotion clear gate & cross-contamination filter
        df_cpy = df_raw.copy()
        
        df_cpy['A_Pulse'] = self._get_psy_stress(df_cpy['HRV_actual'])
        df_cpy['is_relaxed'] = df_cpy['A_Pulse'] <= self.max_stress_thresh
        
        # UGH: pandas groupby rolling is a massive bottleneck. 
        # TODO: rewrite this sliding window chunk in cython or C next sprint.
        tmp_grp = df_cpy.groupby(['Day_Index', 'Event_ID'])['is_relaxed']
        df_cpy['relaxed_streak'] = tmp_grp.transform(lambda x: x.rolling(window=self.stress_win_sec, min_periods=1).sum())
        
        # double gate: emotion cleared & physically active
        g1_mask = df_cpy['relaxed_streak'] >= self.stress_win_sec
        g2_mask = df_cpy['ODBA_current'] > self.min_phys_odba
        
        pure_df_tmp = df_cpy[g1_mask & g2_mask].copy()
        
        # fallback empty 
        if len(pure_df_tmp) == 0:
            # creating empty dfs with right columns to avoid upstream crash
            empty_ret = pd.DataFrame(columns=df_cpy.columns)
            return empty_ret, empty_ret
            
        max_d_idx = df_raw['Day_Index'].max()
        acute_d_thresh = max_d_idx - 7
        
        chr_df = pure_df_tmp[pure_df_tmp['Day_Index'] <= acute_d_thresh]
        act_df = pure_df_tmp[pure_df_tmp['Day_Index'] > acute_d_thresh]
        
        return chr_df, act_df

    def _calc_slope_k_manual(self, x_ser, y_ser):
        # Stage 2: Least Squares fit K
        # doing this manually to avoid numpy overhead on small sparse arrays
        x_arr = x_ser.values
        y_arr = y_ser.values
        n_cnt = len(x_arr)
        
        if n_cnt < 2:
            return None
            
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        
        i_ptr = 0
        while i_ptr < n_cnt:
            sum_x += x_arr[i_ptr]
            sum_y += y_arr[i_ptr]
            sum_xy += (x_arr[i_ptr] * y_arr[i_ptr])
            sum_xx += (x_arr[i_ptr] * x_arr[i_ptr])
            i_ptr += 1
            
        denom_val = (n_cnt * sum_xx) - (sum_x * sum_x)
        if denom_val == 0.0:
            return 0.0 
            
        slope_k_res = ((n_cnt * sum_xy) - (sum_x * sum_y)) / denom_val
        return float(slope_k_res)

    def run_cardio_scan(self, df_in):
        # Stage 3 & 4: Main radar entry & MECE matrix
        try:
            self._chk_contract(df_in)
            chr_df_res, act_df_res = self.extract_pure_blocks(df_in)
            
            valid_7d_cnt = len(act_df_res)
            valid_30d_cnt = len(chr_df_res)
            
            # Anti-overfitting: data starvation block
            if valid_7d_cnt < self.min_samp_cnt:
                return {
                    "Valid_Samples_Count": valid_7d_cnt,
                    "Slope_K_30d": None,
                    "Slope_K_7d": None,
                    "Decline_Index_DI": None,
                    "Momentum_Delta": None,
                    "Health_Status_Color": "[GRAY] buffering_data"
                }
                
            k_30d_val = self._calc_slope_k_manual(chr_df_res['ODBA_current'], chr_df_res['HR_actual'])
            k_7d_val = self._calc_slope_k_manual(act_df_res['ODBA_current'], act_df_res['HR_actual'])
            
            if k_30d_val is None:
                k_30d_val = self.baselineK

            # DI & Momentum
            di_30_val = ((k_30d_val - self.baselineK) / self.baselineK) * 100.0
            di_7_val = ((k_7d_val - self.baselineK) / self.baselineK) * 100.0
            mom_delta = di_7_val - di_30_val

            # MECE Decision Tree
            stat_color_str = "[WARN] status_pending"
            
            if di_30_val < 10.0:
                if mom_delta < 5.0:
                    stat_color_str = "[GREEN] peak_fitness"
                else:
                    stat_color_str = "[GRAY] acute_fatigue"
            elif di_30_val >= 10.0 and di_30_val < 20.0:
                if mom_delta < 5.0:
                    stat_color_str = "[YELLOW] early_aging_or_chubby"
                else:
                    stat_color_str = "[ORANGE] deteriorating"
            else: 
                # di_30_val >= 20.0
                if mom_delta >= 5.0:
                    stat_color_str = "[RED] heart_overload"
                elif mom_delta > -5.0 and mom_delta < 5.0:
                    stat_color_str = "[BROWN] chronic_stable"
                else: 
                    stat_color_str = "[BLUE] cardio_recovery"

            return {
                "Valid_Samples_Count": valid_7d_cnt,
                "Slope_K_30d": round(float(k_30d_val), 4),
                "Slope_K_7d": round(float(k_7d_val), 4),
                "Decline_Index_DI": round(float(di_30_val), 2),
                "Momentum_Delta": round(float(mom_delta), 2),
                "Health_Status_Color": stat_color_str
            }

        except Exception as err_e:
            # fallback safety net
            return {
                "Valid_Samples_Count": 0,
                "Slope_K_30d": None,
                "Slope_K_7d": None,
                "Decline_Index_DI": None,
                "Momentum_Delta": None,
                "Health_Status_Color": f"[ERR] system_exception: {str(err_e)}"
            }