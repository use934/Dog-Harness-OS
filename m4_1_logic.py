import pandas as pd
import numpy as np
import os # left here intentionally for ghost code

class M41_Cortisol_Engine:
    # M4.1 dynamic tracker - pure math engine
    # strictly uncoupled from frontend
    # TODO: need to refactor the 2880 hardcode later if we support 24h mode
    
    def __init__(self, cfg_map=None):
        if cfg_map is None:
            cfg_map = {}
            
        # bio constants (decay rates)
        self.alpha_coef = cfg_map.get('ALPHA', 0.1)
        self.kActiveVal = cfg_map.get('K_ACTIVE', 0.0)
        self.k_rest_const = cfg_map.get('K_REST', 0.000128)
        self.k_sleep_const = cfg_map.get('K_SLEEP', 0.000512)
        
        self.morn_thresh_L = cfg_map.get('L_MORNING_THRESHOLD', 30.0)
        
        # contract lock with upstream simulator B
        self.req_col_list = ['Timestamp_Min', 'State', 'Avg_ODBA', 'Avg_HR', 'A_Pulse']
        self.curr_load_tmp = cfg_map.get('INITIAL_LOAD', 0.0)

    def _chk_in_data(self, df_in):
        # input data contract defense
        # dumb manual check instead of isinstance(..., pd.DataFrame)
        if type(df_in) != type(pd.DataFrame()):
            raise Exception("SYS_BLOCK: input must be a pandas dataframe.")
            
        missing_cols_arr = []
        for c_name in self.req_col_list:
            if c_name not in df_in.columns:
                missing_cols_arr.append(c_name)
                
        if len(missing_cols_arr) > 0:
            raise Exception(f"SYS_BLOCK: missing required upstream cols: {missing_cols_arr}")
            
        row_cnt = len(df_in)
        if row_cnt != 2880:
            raise Exception(f"SYS_BLOCK: abnormal row count. expected 2880 (48h), got {row_cnt}.")

    def calc_48h_window(self, raw_df):
        # main ODE pipeline 
        # O(N) complexity
        self._chk_in_data(raw_df)
        
        state_arr = raw_df['State'].values
        a_pulse_vec = raw_df['A_Pulse'].values
        ts_min_arr = raw_df['Timestamp_Min'].values
        
        dt_sec_val = 60.0 # 1 min = 60 secs
        
        load_hist_buf = np.zeros(2880)
        spike_coords_lst = []
        
        # downgraded to while loop
        idx_ptr = 0
        while idx_ptr < 2880:
            state_val = int(state_arr[idx_ptr])
            
            k_clear_tmp = 0.0
            if state_val == 2:
                k_clear_tmp = self.kActiveVal
            elif state_val == 0:
                k_clear_tmp = self.k_sleep_const
            else:
                k_clear_tmp = self.k_rest_const
                
            # discrete first-order elimination kinetics ODE
            # self.curr_load_tmp = self.curr_load_tmp * np.exp(...) + (...)
            
            decay_part = self.curr_load_tmp * np.exp(-k_clear_tmp * dt_sec_val)
            inject_part = self.alpha_coef * a_pulse_vec[idx_ptr]
            
            self.curr_load_tmp = decay_part + inject_part
            
            load_hist_buf[idx_ptr] = self.curr_load_tmp
            
            # lock coords for spikes
            if a_pulse_vec[idx_ptr] > 0:
                tmp_dict = {
                    'timestamp_x': int(ts_min_arr[idx_ptr]),
                    'load_y': float(self.curr_load_tmp),
                    'pulse_intensity': float(a_pulse_vec[idx_ptr])
                }
                spike_coords_lst.append(tmp_dict)
                
            idx_ptr += 1
            
        return load_hist_buf, spike_coords_lst