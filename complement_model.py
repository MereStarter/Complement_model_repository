import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class ComplementModel:
    def __init__(self):
        self.parameter_names = [
            "k_GZMK_C4", "Km_GZMK_C4", "k_GZMK_C2", "Km_GZMK_C2",
            "n_In2_C4", "n_In2_C2", "alpha_In2_C4", "alpha_In2_C2",
            "n_C5a", "n_C3a", "k_C4bC2a", "k_C3b_basal",
            "k_C3Convertase2", "k_C5Convertase2", "k_C5_C4bC2a_bind",
            "k_C5_conversion", "Km_C5_conversion", "k_C5_conversion_alternate",
            "Km_C5_conversion_alternate", "k_inhibit_C4bC2a", "k_inhibit_C3Convertase2",
            "k_cat_C4bC2a", "Km_C4bC2a", "k_cat_C3convertase2", "Km_C3Convertase2",
            "k_degradationC3a", "kC5inhibit", "k_degradationC5a",
            "k_GZMK_C3", "Km_GZMK_C3", "k_GZMK_C5", "Km_GZMK_C5", 
        ]

    def balances(self, t, x, data_dict):
        """Python implementation of Balances.jl"""
        # Extract species
        GZMK, C4, C2, C4a, C4b, C2a, C2b, C3, C3b, C4bC2a, \
        C3Convertase2, C4bC2aC3b, C5Convertase2, C5, C5a, C5b, \
        C4BP, FactorH, C3a = x

        # Non-negative constraints
        GZMK = max(GZMK, 0)
        C5 = max(C5, 0)
        C3 = max(C3, 0)

        # Get parameters
        params = data_dict["PARAMETER_ARRAY"]
        param_dict = {name: params[i] for i, name in enumerate(self.parameter_names[:-4])}
        param_dict["k_GZMK_C3"] = 0.35*param_dict["k_C4bC2a"]
        param_dict["Km_GZMK_C3"] = 1*param_dict["Km_C4bC2a"]
        param_dict["k_GZMK_C5"] = 1*param_dict["k_C5_conversion"]
        param_dict["Km_GZMK_C5"] = 1*param_dict["Km_C5_conversion"]

        # Reaction rates
        rV = np.zeros(16)
        rV[0] = param_dict["k_GZMK_C4"]*GZMK*C4/(param_dict["Km_GZMK_C4"]+C4)
        rV[1] = param_dict["k_GZMK_C2"]*GZMK*C2/(param_dict["Km_GZMK_C2"]+C2)
        rV[2] = param_dict["k_C4bC2a"]*C4b*C2a
        rV[3] = param_dict["k_C3b_basal"]*C3
        rV[4] = param_dict["k_C3Convertase2"]*C3b
        rV[5] = param_dict["k_C5Convertase2"]*C3Convertase2*C3b
        rV[6] = param_dict["k_C5_C4bC2a_bind"]*C4bC2a*C3b
        rV[7] = param_dict["k_C5_conversion"]*C4bC2aC3b*(C5**param_dict["n_C5a"])/(param_dict["Km_C5_conversion"]**param_dict["n_C5a"]+C5**param_dict["n_C5a"])
        rV[8] = param_dict["k_C5_conversion_alternate"]*C5Convertase2*C5/(param_dict["Km_C5_conversion_alternate"]+C5)
        rV[9] = param_dict["k_inhibit_C4bC2a"]*C4b*C4BP
        rV[10] = param_dict["k_inhibit_C3Convertase2"]*C3Convertase2*FactorH
        rV[11] = param_dict["kC5inhibit"]*C4bC2aC3b*C4BP
        rV[12] = param_dict["k_cat_C4bC2a"]*C4bC2a*(C3**param_dict["n_C3a"])/(param_dict["Km_C4bC2a"]**param_dict["n_C3a"]+C3**param_dict["n_C3a"])
        rV[13] = param_dict["k_cat_C3convertase2"]*C3Convertase2*C3/(param_dict["Km_C3Convertase2"]+C3)
        rV[14] = param_dict["k_GZMK_C3"]*GZMK*C3/(param_dict["Km_GZMK_C3"]+C3)
        rV[15] = param_dict["k_GZMK_C5"]*GZMK*C5/(param_dict["Km_GZMK_C5"]+C5)

        # Control functions
        control_vector = np.ones(2)
        # control_vector[0] = (GZMK**param_dict["n_In2_C4"])/(GZMK**param_dict["n_In2_C4"]+param_dict["alpha_In2_C4"]**param_dict["n_In2_C4"])
        # control_vector[1] = (GZMK**param_dict["n_In2_C2"])/(GZMK**param_dict["n_In2_C2"]+param_dict["alpha_In2_C2"]**param_dict["n_In2_C2"])
        
        # Modified reaction rates
        rV_modified = np.zeros(2)
        rV_modified[0] = rV[0]*control_vector[0]
        rV_modified[1] = rV[1]*control_vector[1]
        
        # Time scaling factor
        tau_C5 = 1.0
        if GZMK > 2e-5:
            tau_C5 = GZMK

        # Material balances
        dxdt = np.zeros(19)
        dxdt[0] = 0  # GZMK
        dxdt[1] = -rV_modified[0]  # C4
        dxdt[2] = -rV_modified[1]  # C2
        dxdt[3] = rV_modified[0]  # C4a
        dxdt[4] = rV_modified[0]-rV[2]  # C4b
        dxdt[5] = rV_modified[1]-rV[2]  # C2a
        dxdt[6] = rV_modified[1]  # C2b
        dxdt[7] = -rV[3]-rV[12]-rV[13]-rV[14]  # C3
        dxdt[8] = rV[3]-rV[4]-rV[5]-rV[6]+rV[12]+rV[13]+rV[14]  # C3b
        dxdt[9] = rV[2]-rV[6]-rV[9]  # C4bC2a
        dxdt[10] = rV[4]-rV[5]-rV[10]  # C3Convertase2
        dxdt[11] = rV[6]-rV[11]  # C4bC2aC3b
        dxdt[12] = rV[5]  # C5Convertase2
        dxdt[13] = -(rV[7]+rV[8]+rV[15])  # C5
        dxdt[14] = tau_C5*(rV[7]+rV[8]+rV[15]-param_dict["k_degradationC5a"]*C5a)  # C5a
        dxdt[15] = tau_C5*(rV[7]+rV[8]+rV[15])  # C5b
        dxdt[16] = -rV[9]-rV[11]  # C4BP
        dxdt[17] = -rV[10]  # FactorH
        dxdt[18] = rV[3]+rV[12]+rV[13]+rV[14]-param_dict["k_degradationC3a"]*C3a  # C3a
        
        return dxdt

    def solve_balances(self, t_start, t_stop, t_step, data_dict):
        """Python implementation of SolveBalances.jl"""
        t_eval = np.arange(t_start, t_stop + t_step, t_step)
        initial_conditions = data_dict["INITIAL_CONDITION_ARRAY"]
        
        sol = solve_ivp(
            fun=lambda t, x: self.balances(t, x, data_dict),
            t_span=(t_start, t_stop),
            y0=initial_conditions,
            t_eval=t_eval,
            method='BDF',
            rtol=1e-4,
            atol=1e-6
        )
        
        return sol.t, sol.y.T

    def run_ensemble(self, data_path="./data"):
        """Python implementation of sample_ensemble.jl"""
        # Load data files
        pc_array = np.loadtxt(f"{data_path}/pc_array_O1_O2.dat")
        rank_array = np.loadtxt(f"{data_path}/rank_array_O1_O2.dat")
        
        # Select top ranked parameters (rank <= 5)
        idx_rank = np.where(rank_array <= 5.0)[0]
        
        # Setup time scale
        t_start, t_stop, t_step = 0.0, 25.0, 0.1
        time_experimental = np.linspace(t_start, t_stop, 200)
        
        # Initialize data structures
        initial_condition_array = np.zeros(19)
        initial_condition_array[0] = 0.1    # GZMK
        initial_condition_array[1] = 1.9    # C4
        initial_condition_array[2] = 0.322  # C2
        initial_condition_array[7] = 7.57   # C3
        initial_condition_array[13] = 0.195 # C5
        initial_condition_array[16] = 2.23  # Factor H
        initial_condition_array[17] = 0.417 # C4BP
        initial_condition_array[18] = 0.6   # C3a initial
        initial_condition_array[14] = 6e-5  # C5a initial

        data_dict = {
            "INITIAL_CONDITION_ARRAY": initial_condition_array,
            "PARAMETER_ARRAY": None
        }

        # Initialize results array
        data_array_C3a = np.zeros((len(time_experimental), 1))
        data_array_C5a = np.zeros((len(time_experimental), 1))
        number_of_samples = 0

        # Run simulation for each parameter set
        for index, rank_index_value in enumerate(idx_rank):
            if index % 1 == 0:
                # Get parameter set
                parameter_array = pc_array[:, rank_index_value]
                data_dict["PARAMETER_ARRAY"] = parameter_array

                # Run model
                t, x = self.solve_balances(t_start, t_stop, t_step, data_dict)

                # Interpolate results to experimental time scale
                f_C3a = interp1d(t, x[:, 18], kind='linear', fill_value="extrapolate")
                f_C5a = interp1d(t, x[:, 14], kind='linear', fill_value="extrapolate")
                IC3a = f_C3a(time_experimental)
                IC5a = f_C5a(time_experimental)

                # Store results
                data_array_C3a = np.column_stack((data_array_C3a, IC3a))
                data_array_C5a = np.column_stack((data_array_C5a, IC5a))
                number_of_samples += 1

        # Calculate statistics
        mean_C3a = np.mean(data_array_C3a[:, 1:], axis=1)
        std_C3a = np.std(data_array_C3a[:, 1:], axis=1)
        mean_C5a = np.mean(data_array_C5a[:, 1:], axis=1)
        std_C5a = np.std(data_array_C5a[:, 1:], axis=1)
        avg_params = np.mean(pc_array[:, idx_rank], axis=1)
        std_params = np.std(pc_array[:, idx_rank], axis=1)
        
        # Load experimental data
        MEASURED_ARRAY_C3a = np.loadtxt(f"{data_path}/Shaw2015_Fig2e_C3a.txt", delimiter=',')
        MEASURED_ARRAY_C5a = np.loadtxt(f"{data_path}/Shaw2015_Fig3c_C5a_original.txt", delimiter=',')
        MC3a = 0.11002 * MEASURED_ARRAY_C3a[:, 1]
        MC5a = 9.615e-5 * MEASURED_ARRAY_C5a[:, 1]
        

        return {
            "time": time_experimental,
            "mean_C3a": mean_C3a,
            "mean_C5a": mean_C5a,
            "std_C3a": std_C3a,
            "std_C5a": std_C5a,
            "samples": number_of_samples,
            "exp_C3a": (MEASURED_ARRAY_C3a[:, 0], MC3a),
            "exp_C5a": (MEASURED_ARRAY_C5a[:, 0], MC5a),
            "avg_params": (avg_params,std_params)
        }
        

if __name__ == "__main__":
    model = ComplementModel()
    results = model.run_ensemble()
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot C3a results
    SF = 2.58  # Scaling factor for 99% CI
    UB_C3a = results["mean_C3a"] + SF * results["std_C3a"]
    LB_C3a = results["mean_C3a"] - SF * results["std_C3a"]
    LB_C3a[LB_C3a < 0] = 0
    
    ax1.fill_between(results["time"], LB_C3a, UB_C3a, color=[0.8, 0.8, 0.8], linewidth=2)
    ax1.plot(results["time"], results["mean_C3a"], "k", linewidth=2)
    ax1.plot(results["exp_C3a"][0], results["exp_C3a"][1],
             color="black", marker="o", markersize=8, linestyle="None")
    ax1.set_xlabel("Time (h)", fontsize=12)
    ax1.set_ylabel("C3a (μM)", fontsize=12)
    ax1.set_title("C3a Dynamics", fontsize=12)
    ax1.grid(True)
    
    # Plot C5a results
    UB_C5a = results["mean_C5a"] + SF * results["std_C5a"]
    LB_C5a = results["mean_C5a"] - SF * results["std_C5a"]
    ax2.fill_between(results["time"], LB_C5a, UB_C5a, color=[0.8, 0.8, 0.8], linewidth=2)
    ax2.plot(results["time"], results["mean_C5a"], "k", linewidth=2)
    ax2.plot(results["exp_C5a"][0], results["exp_C5a"][1],
             color="black", marker="o", markersize=8, linestyle="None")
    ax2.set_xlabel("Time (h)", fontsize=12)
    ax2.set_ylabel("C5a (μM)", fontsize=12)
    ax2.set_title("C5a Dynamics", fontsize=12)
    ax2.grid(True)

    # print(f"Number of samples: {results['samples']}")
    # para_message="params = {"
    # for i in range(len(model.parameter_names)-4):
    #     para_message+=f"{model.parameter_names[i]}-> {results['avg_params'][0][i]}, "
    # para_message=para_message[:-1]+"}"
    # print(para_message)
    
    plt.tight_layout()
    plt.show()