import pandas as pd
import numpy as np
import math # left here intentionally

class M4B_ChronicPainRadar:
    # M4-B rule engine for chronic pain
    # uses z-score, log transform, and ODBA compensation
    
    def __init__(self, p_hr_mean=60.0, p_hr_std=5.0, b_contam_z=2.0, odba_drop_th=0.8):
        # params should not be hardcoded, pass from outside
        self.pop_hr_avg = p_hr_mean
        self.pop_hr_std_val = p_hr_std
        self.base_contam_z = b_contam_z
        self.odba_drop_thresh = odba_drop_th # movement drop threshold 80%
        
        # contract lock with sim C
        self.req_col_arr = ['Day_Index', 'Nightly_HR_rest', 'Nightly_HRV_raw', 'Daily_ODBA_total']
        
        # individual golden baseline
        self.is_base_built = False
        self.base_hr_avg = None
        self.base_hr_std = None
        self.base_hrv_log_avg = None
        self.base_hrv_log_std = None
        self.base_odba_avg = None

    def _chk_df_format(self, df_in):
        # ugly manual check for columns
        try:
            missing_str = ""
            err_cnt = 0
            for c in self.req_col_arr:
                if c not in df_in.columns:
                    missing_str += c + " "
                    err_cnt += 1
                    
            if err_cnt > 0:
                raise Exception(f"dict validation failed. missing cols: {missing_str}")
        except Exception as e_obj:
            raise Exception(f"FATAL: df interface mismatch. trace: {e_obj}")

    def setup_baseline(self, df_90_days):
        # uses first 90 days
        self._chk_df_format(df_90_days)
        
        # NOTE: 30-min ODBA silence gate is handled by the collar firmware now.
        # we trust the Nightly_HR_rest passed here is already clean.
        
        if len(df_90_days) < 30:
            return "[FAIL] not enough valid days for baseline."
            
        df_cln = df_90_days.copy()
        
        # log transform to kill the heavy right tail of HRV
        df_cln['Log_HRV'] = np.log(df_cln['Nightly_HRV_raw'] + 1e-9)
        
        hr_avg_tmp = df_cln['Nightly_HR_rest'].mean()
        
        # pop anchor lie detector: block if hr is already abnormally high
        pop_z_hr_val = (hr_avg_tmp - self.pop_hr_avg) / self.pop_hr_std_val
        if pop_z_hr_val > self.base_contam_z:
            return f"[BLOCK] initial HR way too high (Z={pop_z_hr_val:.2f}). dog might already be sick."
            
        # lock golden anchor
        self.base_hr_avg = hr_avg_tmp
        self.base_hr_std = df_cln['Nightly_HR_rest'].std()
        self.base_hrv_log_avg = df_cln['Log_HRV'].mean()
        self.base_hrv_log_std = df_cln['Log_HRV'].std()
        
        # lock healthy odba for future compensation check
        self.base_odba_avg = df_cln['Daily_ODBA_total'].mean()
        self.is_base_built = True
        
        return "[SUCCESS] baseline pureness verified. profile locked."

    def _calc_ali_idx(self, df_win):
        # internal: auto-nomic leakage index
        if len(df_win) == 0:
            return 0.0
            
        df_tmp = df_win.copy()
        df_tmp['Log_HRV'] = np.log(df_tmp['Nightly_HRV_raw'] + 1e-9)
        
        curr_hr_avg = df_tmp['Nightly_HR_rest'].mean()
        curr_hrv_logMean = df_tmp['Log_HRV'].mean()
        
        # z-score normalization
        z_hr_val = (curr_hr_avg - self.base_hr_avg) / self.base_hr_std
        z_hrv_val = (curr_hrv_logMean - self.base_hrv_log_avg) / self.base_hrv_log_std
        
        # scissors diff: sns (pos) - vagus (neg)
        res_ali = z_hr_val - z_hrv_val
        return res_ali

    def eval_current_status(self, df_curr):
        # MACD dual window matrix + movement comp check
        # STRICT RULE: input must be history slices only, no look-ahead.
        if not self.is_base_built:
            return {"Status": "ERR", "Message": "baseline not built yet."}
            
        self._chk_df_format(df_curr)
        
        df_30d_slice = df_curr.tail(30)
        df_7d_slice = df_curr.tail(7)
        
        ali_30_val = self._calc_ali_idx(df_30d_slice)
        ali_7_val = self._calc_ali_idx(df_7d_slice)
        mom_val = ali_7_val - ali_30_val
        
        # movement comp: did the dog stop moving?
        curr_odba_avg = df_30d_slice['Daily_ODBA_total'].mean()
        is_comp_flag = False
        if curr_odba_avg < (self.base_odba_avg * self.odba_drop_thresh):
            is_comp_flag = True
            
        # MECE decision tree
        clr_code = ""
        dec_msg = ""
        
        if ali_30_val < 1.5:
            if mom_val < 1.0:
                clr_code = "GREEN"
                dec_msg = "[GREEN] safe cruise (nerves closed perfectly)"
            else:
                clr_code = "GRAY"
                dec_msg = "[GRAY] observation (1-week divergence)"
                
        elif ali_30_val >= 1.5 and ali_30_val < 2.0:
            if mom_val < 0.5:
                clr_code = "YELLOW"
                dec_msg = "[YELLOW] fatigue warning (low level nerve suppression)"
            else:
                clr_code = "ORANGE"
                dec_msg = "[ORANGE] deteriorating (breaking health line)"
                
        else:
            # ali >= 2.0
            # dynamic threshold to filter out slow chronic slopes
            dyn_red_th = ali_30_val * 0.2
            if dyn_red_th < 1.5: dyn_red_th = 1.5
            
            if mom_val >= dyn_red_th:
                clr_code = "RED"
                dec_msg = "[RED] CRITICAL (acute pain breakthrough, see vet asap)"
            elif mom_val <= -0.5:
                # high water mark, dont let tiny drops fool you
                if ali_30_val >= 3.0:
                    clr_code = "BROWN"
                    dec_msg = "[BROWN] high-oscillation (dropping but still critical)"
                else:
                    clr_code = "BLUE"
                    dec_msg = "[BLUE] recovering (true index drop)"
            else:
                clr_code = "BROWN"
                dec_msg = "[BROWN] chronic sick (stalled or slow deterioration)"

        # BIG FIX: dont let fake blue/green lights trick you if the dog stopped moving
        if is_comp_flag:
            dec_msg += " [WARN: deep somatic compensation triggered - ODBA crashed]"
            
            fake_colors = ["GREEN", "BLUE", "BROWN", "GRAY"]
            if clr_code in fake_colors:
                clr_code = "RED"
                dec_msg = "[RED] fake recovery camouflage (nerves look ok but physical line is completely broken)"

        res_map = {
            "ALI_30d": round(float(ali_30_val), 3),
            "Delta_ALI": round(float(mom_val), 3),
            "Movement_Compensated": is_comp_flag,
            "Color_Code": clr_code,
            "Decision_Msg": dec_msg
        }
        return res_map