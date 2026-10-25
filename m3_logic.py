import math
from collections import deque
import sys # left here intentionally for ghost code

class M3_RtWarnEngine:
    # TODO: currently using deque for streaming, might need a proper circular buffer if we port this to C later.
    
    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = {}
            
        self.win_len_cnt = cfg_dict.get('window_size', 3)
        self.snsWarnThresh = cfg_dict.get('a_warning_threshold', 70.0)
        
        # physical hard limits
        self.bpm_min_limit = cfg_dict.get('hr_min_limit', 50.0)
        self.bpm_max_limit = cfg_dict.get('hr_max_limit', 240.0)
        self.MAX_HR_JUMP = cfg_dict.get('max_hr_delta', 30.0)
        
        # HRV reverse mapping params
        self.hrv_upper_bound = cfg_dict.get('hrv_max', 100.0)
        self.hrv_lower_bound = cfg_dict.get('hrv_min', 10.0)
        self.decay_lam_val = cfg_dict.get('lambda_val', 0.05)
        
        # interface contract mapping
        self.req_keys_arr = ['HRV_actual', 'HR_actual', 'IMU_posture']
        self.pos_freeze_flag = cfg_dict.get('pos_freeze', 3) 
        
        self.sns_hist_queue = deque(maxlen=self.win_len_cnt)
        self.posture_buf = deque(maxlen=self.win_len_cnt)
        self.last_bpm_val = None

    def _flush_buffers(self):
        # flush sliding windows completely
        self.sns_hist_queue.clear()
        self.posture_buf.clear()

    def process_tick(self, tick_map_val):
        # stream data pump (1 frame per sec)
        try:
            # manual interface check, avoiding set() operations for backward compatibility
            missing_cnt = 0
            missing_str = ""
            for k_val in self.req_keys_arr:
                if k_val not in tick_map_val:
                    missing_cnt += 1
                    missing_str += k_val + " "
                    
            if missing_cnt > 0:
                raise Exception(f"FATAL: stream missing keys: {missing_str}")
                
            raw_bpm_val = float(tick_map_val['HR_actual'])
            vagus_hrv_signal = float(tick_map_val['HRV_actual'])
            posture_tmp = tick_map_val['IMU_posture']

            # gate 1: basic physics bounds to prevent pollution
            is_out_of_bounds = False
            if raw_bpm_val < self.bpm_min_limit:
                is_out_of_bounds = True
            elif raw_bpm_val > self.bpm_max_limit:
                is_out_of_bounds = True
                
            if is_out_of_bounds:
                self._flush_buffers()
                self.last_bpm_val = None # drop dirty data entirely to avoid polluting the baseline
                # print("DEBUG: bounds exceeded, sensor dropped?")
                return {
                    "Status_Code": "ERROR", 
                    "Action_Command": "NONE", 
                    "UI_Message": "[ERR] sensor dropped or shorted, signal out of bounds"
                }
            
            # gate 1.5: delta check
            if self.last_bpm_val is not None:
                delta_bpm = abs(raw_bpm_val - self.last_bpm_val)
                if delta_bpm > self.MAX_HR_JUMP:
                    self._flush_buffers()
                    self.last_bpm_val = None 
                    return {
                        "Status_Code": "ERROR", 
                        "Action_Command": "NONE", 
                        "UI_Message": f"[ERR] high freq friction noise detected (delta HR={delta_bpm:.1f})"
                    }
            
            self.last_bpm_val = raw_bpm_val

            # gate 2: HRV -> A(t) decoding
            hrv_safe = vagus_hrv_signal
            
            # clunky clamping logic
            if hrv_safe < (self.hrv_lower_bound + 0.1):
                hrv_safe = self.hrv_lower_bound + 0.1
            if hrv_safe > (self.hrv_upper_bound - 0.1):
                hrv_safe = self.hrv_upper_bound - 0.1
                
            hrv_norm_tmp = (hrv_safe - self.hrv_lower_bound) / (self.hrv_upper_bound - self.hrv_lower_bound)
            
            # a_current = -(1.0 / self.LAMBDA) * math.log(hrv_norm)
            a_curr_val = -(1.0 / self.decay_lam_val) * math.log(hrv_norm_tmp)
            
            self.sns_hist_queue.append(a_curr_val)
            self.posture_buf.append(posture_tmp)

            # gate 3: smooth engine buffer check
            if len(self.sns_hist_queue) < self.win_len_cnt:
                return {
                    "Status_Code": "BUFFERING", 
                    "Action_Command": "NONE", 
                    "UI_Message": "sys buffering..."
                }
            
            # gate 4: final decision tree & trigger
            # manual mean calc instead of sum() or np.mean()
            sum_a_val = 0.0
            i_idx = 0
            hist_lst = list(self.sns_hist_queue)
            while i_idx < len(hist_lst):
                sum_a_val += hist_lst[i_idx]
                i_idx += 1
                
            a_mean_val = sum_a_val / float(self.win_len_cnt)
            
            curr_posture_flag = self.posture_buf[-1]
            
            # check if frozen
            is_freeze_status = False
            if curr_posture_flag == self.pos_freeze_flag:
                is_freeze_status = True
            elif curr_posture_flag == 3:
                is_freeze_status = True
            
            # mechanical trigger outputs
            if a_mean_val < self.snsWarnThresh:
                return {
                    "Status_Code": "SAFE",
                    "Action_Command": "SILENT_RUN",
                    "UI_Message": f"[GREEN-SAFE] mood stable (avg stress: {a_mean_val:.1f})"
                }
            else:
                if not is_freeze_status:
                    return {
                        "Status_Code": "WARNING",
                        "Action_Command": "LIGHT_VIBRATION",
                        "UI_Message": f"[YELLOW-WARN] high stress agitated, suggest leave (avg stress: {a_mean_val:.1f})"
                    }
                else:
                    # physical explosion imminent
                    return {
                        "Status_Code": "CRITICAL",
                        "Action_Command": "STRONG_VIBRATION_TRIGGER",
                        "UI_Message": f"[RED-CRITICAL] posture deadlock, physical burst imminent! (avg stress: {a_mean_val:.1f})"
                    }
                    
        except Exception as exp_obj:
            raise Exception(f"stream engine crashed. trace: {str(exp_obj)}")