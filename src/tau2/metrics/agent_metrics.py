import math
import re
import os
import sys
import pandas as pd
from loguru import logger
from pydantic import BaseModel
import json
from tau2.data_model.simulation import Results
from tau2.utils.version_profiler import get_version_info
from pathlib import Path

def is_successful(reward: float) -> bool:
    """
    Check if the reward is successful.
    """
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


class AgentMetrics(BaseModel):
    avg_reward: float
    pass_hat_ks: dict[int, float]
    avg_agent_cost: float

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_agent_cost": self.avg_agent_cost,
        }
        for k, v in self.pass_hat_ks.items():
            data[f"pass_hat_{k}"] = v
        return data


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def get_metrics_df(results: Results) -> tuple[pd.DataFrame, int]:
    """
    Convert the results to a dataframe and add a column for success.
    Checks that all simulations have the same number of trials.
    Returns the maximum number of trials that can be used for pass^k metrics.
    """
    df = results.to_df()
    df["success"] = df.reward.apply(is_successful)
    if len(df.info_num_trials.unique()) > 1:
        logger.warning(
            f"All simulations must have the same number of trials. Found {df.info_num_trials.unique()}"
        )
    max_k = df.info_num_trials.max()

    task_ids_counts = [(tid, count) for tid, count in df.task_id.value_counts().items()]
    task_ids_counts.sort(key=lambda x: x[1])
    min_k = task_ids_counts[0][1]
    if min_k < max_k:
        logger.warning(
            f"The minimum number of trials for a task is {min_k}, which is less than the expected number of trials {max_k}. Setting max k to {min_k}."
        )
        max_k = min_k
    return df, max_k


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    dfs = []
    for k in range(1, max_k + 1):
        res = df.groupby("task_id")["success"].apply(
            lambda df: pass_hat_k(len(df), df.sum(), k)
        )
        res.name = f"pass^{k}"
        dfs.append(res)
    df_pass_hat_k = pd.concat(dfs, axis=1)
    task_columns = [
        "task_num_agent_actions",
        "task_num_user_actions",
        "task_num_actions",
    ]
    df_task_infos = df.groupby("task_id").first()[task_columns]
    df_pass_hat_k = df_task_infos.join(df_pass_hat_k)
    return df_pass_hat_k


def prepare_dfs(results: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    df_pass_hat_k["num_actions"] = df.groupby("task_id").first()["task_num_actions"]
    df_pass_hat_k = df_pass_hat_k.sort_values(by="num_actions")
    return df, df_pass_hat_k


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute metrics for the agent.
    - average reward
    - pass^k
    """
    df, df_pass_hat_k = prepare_dfs(results)
    avg_reward = df.reward.mean()
    pass_hat_ks = {}
    for column in df_pass_hat_k.columns:
        if match := re.match(r"pass\^(\d+)", column):
            k = int(match.group(1))
            pass_hat_ks[k] = df_pass_hat_k[column].mean()
    avg_agent_cost = df.agent_cost.mean()
    return AgentMetrics(
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        avg_agent_cost=avg_agent_cost,
    )


def save_scores(metrics: AgentMetrics) -> None:
    model = os.getenv("BASE_MODEL_NAME")
    model_path = os.getenv("MODEL_PATH")
    if "-instruct" in model_path or "-reasoning" in model_path:
        model = model_path.split("/")[-1]
    domain = os.getenv("TEST_DOMAIN")
    job_id = os.getenv("SLURM_JOBID")
    result_folder = os.getenv("RESULT_PATH")
    
    # Check if all required environment variables are set
    if not all([model, domain, job_id, result_folder]):
        missing_vars = []
        if not model:
            missing_vars.append("BASE_MODEL_NAME")
        if not domain:
            missing_vars.append("TEST_DOMAIN")
        if not job_id:
            missing_vars.append("SLURM_JOBID")
        if not result_folder:
            missing_vars.append("RESULT_PATH")
        error_msg = f"Cannot save scores: missing environment variables: {', '.join(missing_vars)}"
        logger.warning(error_msg)
        print(f"⚠️  {error_msg}", file=sys.stderr)
        return
    
    # Create result directory if it doesn't exist
    os.makedirs(result_folder, exist_ok=True)
    
    result_file = os.path.join(result_folder, f"tau2_bench_{domain}.json")
    try:
        metrics_dict = metrics.as_dict()
        score = metrics_dict["avg_reward"]
        del metrics_dict["avg_reward"]
        metrics_dict["overall_score"] = score
        final_result = {}
        model_name = os.getenv(f"MODEL_PATH")
        final_result["model_name"] = model_name
        final_result["domain"] = domain
        final_result["job_id"] = job_id
        final_result["eval_name"] = "tau2_bench_" + domain
        final_result["final_score"] = score
        final_result["metrics"] = metrics_dict
        version_info = get_version_info(
            tau2_root=None,
            env_path=os.getenv("CONDA_PREFIX")
        )
        
        final_result["_VERSION_INFO_"] = version_info
        with open(result_file, "w") as f:
            json.dump(final_result, f, indent=2)
        success_msg = f"Saved metrics to {result_file}"
        logger.info(success_msg)
        print(f"✅ {success_msg}")
    except Exception as e:
        error_msg = f"Failed to save scores to {result_file}: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}", file=sys.stderr)

def display_metrics(metrics: AgentMetrics) -> None:
    print(f"🏆 Average reward: {metrics.avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in metrics.pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
    print(f"💰 Average agent cost: {metrics.avg_agent_cost}")
    save_scores(metrics)




if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    results = Results.load(Path(args.results))
    metrics = compute_metrics(results)
    display_metrics(metrics)
