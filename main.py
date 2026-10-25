import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time # left here intentionally

# ==========================================
# Module Imports (Aligned with de-AI'd contracts)
# ==========================================
try:
    # simulators
    from partA_Simulator import generate_sim_incident_block
    from partB_Simulator import generate_metabolic_block
    from partC_Simulator import generate_chronic_sim_data
    from partD_simulator import generate_sparse_d_block
    
    # logic engines
    from m1_logic import M1_MotivationCore
    from m3_logic import M3_RtWarnEngine
    from m4_1_logic import M41_Cortisol_Engine
    from m4_3_logic import M4B_ChronicPainRadar
    from m4_5_logic import M45_PredictiveEngine
    from m2_logic import M2_TemperamentEngine
    from m4_4_logic import M4C_CardioRadar
except ImportError as e_obj:
    print(f"\n[FATAL] module import failed. check filenames. trace: {e_obj}")
    sys.exit(1)

# ==========================================
# Core Routing & Viz
# ==========================================
def plot_m45_cone_chart(df_full_history, curr_day_idx, m45_res_map, pred_engine_obj, fut_days_cnt=40):
    # renders the logistic prediction cone and god-mode future truth
    cone_data = m45_res_map.get('pred_cone_data')
    if not cone_data:
        # print("DEBUG: no cone data to plot")
        return 
        
    b_params = cone_data['p_dict']
    w_case = cone_data['worst_c']
    b_case = cone_data['best_c']

    hr_arr = df_full_history['Nightly_HR_rest'].values
    hrv_arr = df_full_history['Nightly_HRV_raw'].values
    
    # call internal method, kinda hacky but works for dev mode
    ali_arr = pred_engine_obj._get_ali_array(hr_arr, hrv_arr)
    day_idx_arr = df_full_history['Day_Index'].values
    
    # x axis extension
    x_pred_arr = np.arange(day_idx_arr[0], curr_day_idx + fut_days_cnt)

    def _eval_logist(x_in, l_val, k_val, x0_val):
        # clip to prevent exp overflow on matplotlib canvas
        pwr_tmp = np.clip(-k_val * (x_in - x0_val), -100, 100)
        return l_val / (1.0 + np.exp(pwr_tmp))

    y_base_val = _eval_logist(x_pred_arr, b_params['L'], b_params['k'], b_params['x0'])
    y_worst_val = _eval_logist(x_pred_arr, w_case['L'], w_case['k'], w_case['x0'])
    y_best_val = _eval_logist(x_pred_arr, b_case['L'], b_case['k'], b_case['x0'])

    # TODO: matplotlib blocking the main thread, need to move to async queue later
    plt.figure(figsize=(12, 6))
    
    # boolean masks broken down
    v_mask = ~np.isnan(ali_arr)
    past_mask = np.logical_and(v_mask, day_idx_arr <= curr_day_idx)
    fut_mask = np.logical_and(v_mask, day_idx_arr > curr_day_idx)
    
    plt.scatter(day_idx_arr[past_mask], ali_arr[past_mask], color='gray', alpha=0.5, label='Hist Data (Engine Visible)')
    plt.scatter(day_idx_arr[fut_mask], ali_arr[fut_mask], facecolors='none', edgecolors='green', marker='o', alpha=0.8, linewidth=1.5, label='Future Truth (Blind)')
    
    plt.plot(x_pred_arr, y_base_val, color='red', linewidth=2, label='Base Pred')
    plt.fill_between(x_pred_arr, y_best_val, y_worst_val, color='red', alpha=0.15, label='Conf Cone (Best-Worst)')

    plt.axvline(x=curr_day_idx, color='blue', linestyle='--', linewidth=1.5, label=f'Current Day ({curr_day_idx})')
    
    plt.title("M4.5 Chronic Predictor - Dev Validation", fontsize=14, fontweight='bold')
    plt.xlabel("Day Index", fontsize=12)
    plt.ylabel("ALI (Auto-nomic Leakage)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def run_test_block_a():
    print("\n" + "="*50)
    print("▶ EXEC: Block A (Transient Crisis - M1, M3)")
    print("="*50)
    
    scen_input = input("Enter test scen [default 'malicious_provocation']: ")
    if scen_input == "": scen_input = "malicious_provocation"
    
    df_simA, truth_lbl = generate_sim_incident_block(scen_str=scen_input)
    print(f"\n[SYS_OK] generated {len(df_simA)} rows 1Hz transient data.")
    print(f"[TRUTH] {truth_lbl}")
    
    print("\n--- Init M1 Core (180s Blackbox) ---")
    m1_core = M1_MotivationCore()
    m1_res = m1_core.run_analysis(df_simA)
    print(f"[M1_OUT] T0 stamp: {m1_res.get('t0_stamp')}s")
    print(f"         Motive: {m1_res.get('motive_lbl')}")
    print(f"         Resid_S: {m1_res.get('resid_stress')}")
    
    print("\n--- Init M3 Engine (Rt Stream Test) ---")
    
    # clumsy dict extraction
    rec_head = df_simA.iloc[0:3].to_dict(orient='records')
    rec_tail = df_simA.iloc[-3:].to_dict(orient='records')
    
    m3_inst_head = M3_RtWarnEngine()
    print("[M3_TEST] Pumping first 3 secs (safe):")
    out_dict_h = {}
    for r_obj in rec_head:
        out_dict_h = m3_inst_head.process_tick(r_obj)
    print(f" -> head out: {out_dict_h}")

    m3_inst_tail = M3_RtWarnEngine()
    print("[M3_TEST] Pumping last 3 secs (explosion):")
    out_dict_t = {}
    for r_obj in rec_tail:
        out_dict_t = m3_inst_tail.process_tick(r_obj)
    print(f" -> tail out: {out_dict_t}")

def run_test_block_b():
    print("\n" + "="*50)
    print("▶ EXEC: Block B (Metabolic Continuum - M4.1)")
    print("="*50)
    
    scen_input = input("Enter test scen [default 'cross_day_stress_residue']: ")
    if scen_input == "": scen_input = "cross_day_stress_residue"
    
    df_simB, t_lbl = generate_metabolic_block(scen_str=scen_input)
    print(f"\n[SYS_OK] generated {len(df_simB)} mins continuum data.")
    print(f"[TRUTH] {t_lbl}")
    
    print("\n[SLICE] Top 3 stress peaks:")
    # breaking down pandas chain
    df_lg = df_simB.nlargest(3, 'A_Pulse')
    print(df_lg[['Timestamp_Min', 'State', 'Avg_ODBA', 'Avg_HR', 'A_Pulse']])
    
    print("\n--- Init M4.1 Cortisol Tracker ---")
    trk_eng = M41_Cortisol_Engine(cfg_map={})
    
    ld_hist, spk_cords = trk_eng.calc_48h_window(df_simB)
    
    print(f"[M4.1_OUT] Extracted {len(ld_hist)} mins load array.")
    print(f"           Captured {len(spk_cords)} pulses. Rendering plot...")
    
    plt.figure(figsize=(14, 6))
    plt.plot(df_simB['Timestamp_Min'], ld_hist, color='#1f77b4', linewidth=2, label='Load L(t)')
    
    for spk in spk_cords:
        plt.scatter(spk['timestamp_x'], spk['load_y'], color='red', s=80, zorder=5, edgecolors='black')
        plt.annotate(f"P: {spk['pulse_intensity']}\nL: {spk['load_y']:.1f}", 
                     (spk['timestamp_x'], spk['load_y']),
                     textcoords="offset points", 
                     xytext=(0, 15), 
                     ha='center', fontsize=10, color='red', fontweight='bold')
        
    plt.title('M4.1 Dev Vis - 48H Cortisol Tracker', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Mins)', fontsize=12)
    plt.ylabel('Load L(t)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 23:00 - 07:00 sleep zones
    plt.axvspan(0, 420, color='gray', alpha=0.1, label='Deep Sleep/Clearance')
    plt.axvspan(1380, 1860, color='gray', alpha=0.1)
    plt.axvspan(2820, 2880, color='gray', alpha=0.1)
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def run_test_block_c():
    print("\n" + "="*50)
    print("▶ EXEC: Block C (Chronic Lifecycle - M4.3, M4.5)")
    print("="*50)
    
    print("Select C scen:")
    print("  [1] baseline")
    print("  [2] linear_drift")
    print("  [3] logistic_collapse (default)")
    print("  [4] allostatic_compensation")
    
    c_input = input("Enter ID [1-4]: ").strip()
    
    s_map = {
        "1": "baseline",
        "2": "linear_drift",
        "3": "logistic_collapse",
        "4": "allostatic_compensation"
    }
    scen_str = s_map.get(c_input, "logistic_collapse")

    t_age = 10.0  
    t_wgt = 30.0 
    
    df_chr, t_map = generate_chronic_sim_data(
        scenarioStr=scen_str, 
        ageYrs=t_age, 
        weightKg=t_wgt, 
        tot_days=180, 
        r_seed=None
    )
    print(f"\n[SYS_OK] generated {len(df_chr)} rows lifecycle data.")
    print(f"[TRUTH] {t_map}")
    print("\n[SLICE] Last 3 days:")
    print(df_chr.tail(3))
    
    print("\n--- Init M4.3 (M4-B) Radar ---")
    m4b_eng = M4B_ChronicPainRadar()
    
    df_base = df_chr.iloc[:90]
    b_msg = m4b_eng.setup_baseline(df_base)
    print(f"[M4.3_BASE] {b_msg}")
    
    if not m4b_eng.is_base_built:
        print("[FATAL] golden baseline failed, skipping m4.3 dynamic test.")
    else:
        print("\n[M4.3_RUN] sweeping day 91 to end...")
        prev_clr = None
        
        for cur_d in range(91, len(df_chr) + 1):
            df_sl = df_chr.iloc[:cur_d]
            m_res = m4b_eng.eval_current_status(df_sl)
            
            curr_c = m_res.get('Color_Code')
            if curr_c != prev_clr or cur_d == len(df_chr):
                a_val = m_res.get('ALI_30d')
                d_val = m_res.get('Delta_ALI')
                m_str = m_res.get('Decision_Msg')
                print(f" -> [Day {cur_d}] State Shift: [{curr_c}] | ALI: {a_val} | Mom: {d_val} | {m_str}")
                prev_clr = curr_c

    print("\n--- Init M4.5 Predictor ---")
    # force std_hr=2.0 to unlock signal
    m45_eng = M45_PredictiveEngine(cfg_map={"std_hr": 2.0})
    
    cut_d = 105 
    df_sl_2 = df_chr.iloc[:cut_d]
    
    p_res = m45_eng.process_time_window(df_sl_2, age_y=t_age, weight_k=t_wgt)
    
    print(f"[M4.5_OUT] Alarm: {p_res.get('alarm_state')}")
    print(f"           L3 Trigger: {p_res.get('trigger_l3')}")
    
    if p_res.get('pred_cone_data'):
        print("           [SYS_OK] Cone captured. Rendering visualization...")
        plot_m45_cone_chart(df_chr, cut_d, p_res, m45_eng, fut_days_cnt=40)
    else:
        print("           Cone: None (No collapse trend found or diverged)")

def run_test_block_d():
    print("\n" + "="*50)
    print("▶ EXEC: Block D (Sparse Event Stream - M2, M4.4)")
    print("="*50)
    
    scen_input = input("Enter test scen [default 'obese_and_neurotic']: ")
    if scen_input == "": scen_input = "obese_and_neurotic"
    
    df_d, truth_map = generate_sparse_d_block(scen_str=scen_input)
    print(f"\n[SYS_OK] generated {len(df_d)} rows high freq sparse blocks.")
    print(f"[TRUTH] {truth_map}")
    
    print("\n--- Init M2 Temperament Engine ---")
    m2_eng = M2_TemperamentEngine()
    avg_th, m2_stat = m2_eng.eval_30d_base(df_d)
    
    th_str = f"{avg_th:.2f}s" if avg_th is not None else "None"
    print(f"[M2_OUT] T_half (tau): {th_str}")
    print(f"         Sys State: {m2_stat}")
    
    print("\n--- Init M4.4 (M4-C) Cardio Radar ---")
    m4c_eng = M4C_CardioRadar()
    c_res = m4c_eng.run_cardio_scan(df_d)
    
    print(f"[M4.4_OUT] Valid pure samples: {c_res.get('Valid_Samples_Count', 0)}")
    
    k30 = c_res.get('Slope_K_30d')
    k7 = c_res.get('Slope_K_7d')
    di = c_res.get('Decline_Index_DI')
    mom = c_res.get('Momentum_Delta')
    
    print(f"           K_30d: {k30:.4f}" if k30 is not None else "           K_30d: None")
    print(f"           K_7d:  {k7:.4f}" if k7 is not None else "           K_7d:  None")
    print(f"           DI:    {di:.2f}%" if di is not None else "           DI:    None")
    print(f"           Mom:   {mom:.2f}%" if mom is not None else "           Mom:   None")
    print(f"           Final Alert: {c_res.get('Health_Status_Color')}")

# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n" + "#"*50)
        print(" Canine Health Radar - Dev Console (Build 2.1)")
        print("#"*50)
        print(" [A] Test Transient Defense (M1, M3)")
        print(" [B] Test Metabolic Cortisol (M4.1)")
        print(" [C] Test Chronic Predictor (M4.3, M4.5)")
        print(" [D] Test Sparse Stream Filter (M2, M4.4)")
        print(" [Q] Quit Sys")
        print("-" * 50)
        
        sel = input("Select channel [A/B/C/D/Q]: ").strip().upper()
        
        if sel == 'A':
            run_test_block_a()
        elif sel == 'B':
            run_test_block_b()
        elif sel == 'C':
            run_test_block_c()
        elif sel == 'D':
            run_test_block_d()
        elif sel == 'Q':
            print("sys exit 0.")
            break
        else:
            print("[ERR] invalid input.")