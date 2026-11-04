import gurobipy as gp
import pathlib
import pandas as pd

root = pathlib.Path(__file__).resolve().parents[1]

from system import System, UNSERVED_TOL

def simulate_ols(system, seeds_and_dates, year=2020, save_dispatch=True):
    """Re-solve the given (scenario_seed, date) pairs on `system`.

    Returns one row per pair: unserved energy cost, total unserved energy, solver status.
    """
    results_dir = root / "results" / system.system_config
    results_dir.mkdir(exist_ok=True, parents=True)
    dispatch_dir = results_dir / "dispatch-results"
    dispatch_dir.mkdir(exist_ok=True, parents=True)

    system.read_timeseries(year)
    system.write_opf()

    rows = []
    for _, row in seeds_and_dates.iterrows():
        scenario_seed = int(row.scenario_seed)
        date = row.date

        system.scenario_seed = scenario_seed
        for resource in system.resources.values():
            resource.get_outage_profile(year, scenario_seed)

        system.solve_opf(date)
        if system.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"seed {scenario_seed}, {date.date()}: "f"solver status {system.model.Status}")

        # Save optimal cost, unserved energy, optimality status
        ue_profile = system.unserved_energy_profile
        load_shed_profile = system.load_shed_profile
        rows.append({
            "scenario_seed": scenario_seed,
            "date": date,
            "cost": float(system.total_unserved_energy_costs.getValue()),
            "unserved_energy": float(ue_profile.values.sum()),
            "opt_status": system.model.Status,
        })

        # Save dispatch results
        if save_dispatch:
            system.dispatch_results.to_csv(dispatch_dir / f"dispatch_results_seed{scenario_seed}_{date:%Y-%m-%d}.csv")
            load_shed_profile.to_csv(dispatch_dir / f"load_shed_results_seed{scenario_seed}_{date:%Y-%m-%d}.csv")
            ue_profile.to_csv(dispatch_dir / f"unserved_energy_results_seed{scenario_seed}_{date:%Y-%m-%d}.csv")

    return pd.DataFrame(rows).set_index(["scenario_seed", "date"])

if __name__ == "__main__":

    system_config = "RTS-ORB"
    n_years = 100

    # Read in results
    results_dir = root / "results" / system_config
    df_sim_results = pd.read_csv(results_dir / f"simulation_results_{n_years}yrs.csv", parse_dates=["date"])
    columns = ["scenario_seed", "date", "total_ue", "ue_costs", "event_type"]
    seeds_and_dates = df_sim_results.loc[df_sim_results.total_ue > UNSERVED_TOL, columns]

    system = System(root / "data", root / "system-config", system_config, dynamic_voll=True)
    df_ols_results = simulate_ols(system, seeds_and_dates)

    # Compare against the static-VoLL LP these days were selected from
    df_lp = seeds_and_dates.set_index(["scenario_seed", "date"])
    df_ols_results["lp_unserved_energy"] = df_lp["total_ue"]
    df_ols_results["lp_cost"] = df_lp["ue_costs"]
    df_ols_results["event_type"] = df_lp["event_type"]

    print(df_ols_results.to_string())
    df_ols_results.to_csv(results_dir / f"ols_results_{n_years}yrs.csv")
