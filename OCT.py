"""
LAM-only architecture baseline for the revised mixed routine/complex workload.

All requests are processed by a real LAM service. The workload generation and
sandbox evaluator are aligned with the revised DT-only baseline.

Revisions:
1. The workload uses an exact 60% / 20% / 20% composition and is shuffled.
2. The LAM prompt does not receive the artificial task_type label. It must infer
   single-link congestion, multi-link coordination, and policy constraints from
   the actual network state and constraint fields.
3. LAM inference latency, sandbox-verification latency, and total service
   latency are measured separately.
4. DeepSeek Thinking Mode remains disabled.
5. The final summary reports only five architecture-level comparison metrics.

Run:
    python lam_only_mixed_tasks_revised.py
"""

import copy
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ============================================================
# 1. Real LAM client
# ============================================================

client = OpenAI(
    api_key="sk-73e3fc16c9104b0ab5d26a809f313d5f",
    base_url="https://api.deepseek.com"
)

MODEL_NAME = "deepseek-v4-flash"

# ============================================================
# 2. Experimental parameters
# ============================================================

# Keep this identical to the revised DT-only experiment for direct comparison.
NUM_TASKS = 20
RANDOM_SEED = 2026

ROUTINE_TASK_RATIO = 0.60
MULTI_LINK_TASK_RATIO = 0.20
POLICY_TASK_RATIO = 0.20

OVERLOAD_THRESHOLD = 0.90
LOAD_PERTURBATION = 0.05
SANDBOX_SCENARIOS = 1000
ROBUST_PASS_RATIO = 0.95
SAFETY_MARGIN_MBPS = 0.50
FLOAT_TOLERANCE = 1e-6
MAX_TASK_GENERATION_ATTEMPTS = 2000

MAX_API_RETRIES = 3
RETRY_INTERVAL_SECONDS = 2.0
PRINT_PROMPT = True


# ============================================================
# 3. Data structures
# ============================================================

@dataclass
class NetworkTask:
    task_id: int

    # Used only for experiment grouping/statistics.
    # This field is NOT included in the LAM prompt.
    task_type: str

    capacities: Dict[str, float]
    loads: Dict[str, float]
    traffic_budget_mbps: float
    max_actions: int
    forbidden_target_links: List[str]
    complexity: float

    # Diagnostic ground-truth fields. They are NOT provided to the LAM.
    feasible_sources: List[str]
    feasible_targets: List[str]
    robust_required_traffic: Dict[str, float]


@dataclass
class CandidateAction:
    action: str
    from_link: str
    to_link: str
    traffic: float


@dataclass
class CandidatePlan:
    actions: List[CandidateAction]


@dataclass
class LAMMeasurement:
    plan: Optional[CandidatePlan]
    ttft_ms: float
    total_latency_ms: float
    output_text: str
    api_success: bool
    parse_success: bool
    error_message: str


@dataclass
class EffectMetrics:
    plan_generated: bool
    structurally_valid: bool
    nominal_passed: bool
    robust_passed: bool

    before_max_utilization: float
    after_max_utilization: float
    max_utilization_reduction: float

    overloaded_links_before: int
    overloaded_links_after: int
    overloaded_link_reduction: int

    sandbox_pass_ratio: float
    feedback: str


@dataclass
class ExperimentResult:
    architecture: str
    task_id: int
    task_type: str
    task_complexity: float

    lam_inference_latency_ms: float
    verification_latency_ms: float
    service_latency_ms: float
    ttft_ms: float

    api_success: int
    parse_success: int

    strategy_generated: int
    unresolved_task: int
    invalid_generated_strategy: int
    nominal_success: int
    robust_success: int

    before_max_utilization: float
    after_max_utilization: float
    max_utilization_reduction: float

    overloaded_links_before: int
    overloaded_links_after: int
    overloaded_link_reduction: int

    sandbox_pass_ratio: float

    action_count: int
    total_moved_traffic_mbps: float
    traffic_budget_mbps: float

    lam_invoked: int
    lam_calls: int

    action_plan: str
    raw_lam_output: str
    error_message: str


# ============================================================
# 4. Shared utility functions and exact mixed workload generation
# ============================================================

def ceil_to_two_decimals(value: float) -> float:
    return math.ceil((value - FLOAT_TOLERANCE) * 100.0) / 100.0


def utilization(load: float, capacity: float) -> float:
    if capacity <= 0:
        raise ValueError("Link capacity must be positive.")
    return load / capacity


def state_metrics(
    loads: Dict[str, float],
    capacities: Dict[str, float],
) -> Tuple[float, int]:
    values = {
        link: utilization(loads[link], capacities[link])
        for link in loads
    }

    maximum = max(values.values())

    overloaded = sum(
        value > OVERLOAD_THRESHOLD + FLOAT_TOLERANCE
        for value in values.values()
    )

    return maximum, overloaded


def required_source_traffic(
    source_load: float,
    source_capacity: float,
    perturbation: float = 0.0,
) -> float:
    worst_source_load = source_load * (1.0 + perturbation)

    return max(
        0.0,
        worst_source_load
        - OVERLOAD_THRESHOLD * source_capacity,
    )


def target_headroom(
    target_load: float,
    target_capacity: float,
    perturbation: float = 0.0,
) -> float:
    worst_target_load = target_load * (1.0 + perturbation)

    return max(
        0.0,
        OVERLOAD_THRESHOLD * target_capacity
        - worst_target_load,
    )


def plan_to_string(
    plan: Optional[CandidatePlan],
) -> str:
    if plan is None:
        return "None"

    serializable = []

    for action in plan.actions:
        serializable.append(
            {
                "action": action.action,
                "from_link": action.from_link,
                "to_link": action.to_link,
                "traffic": action.traffic,
            }
        )

    return str(serializable)


# ============================================================
# 4. Mixed workload generation
# ============================================================

def choose_task_type(rng: random.Random) -> str:
    value = rng.random()

    if value < ROUTINE_TASK_RATIO:
        return "routine_single_link"

    if value < ROUTINE_TASK_RATIO + MULTI_LINK_TASK_RATIO:
        return "complex_multi_link"

    return "complex_policy_constraint"


def generate_routine_task(
    task_id: int,
    rng: random.Random,
) -> NetworkTask:
    """
    Generate a single-link congestion task with a robust one-action solution.
    """
    capacities = {
        "L1": 100.0,
        "L2": 100.0,
        "L3": 100.0,
        "L4": 100.0,
    }

    for _ in range(MAX_TASK_GENERATION_ATTEMPTS):
        source = rng.choice(list(capacities.keys()))

        loads = {
            link: round(rng.uniform(28.0, 60.0), 2)
            for link in capacities
        }

        loads[source] = round(
            rng.uniform(94.0, 108.0),
            2,
        )

        robust_required = ceil_to_two_decimals(
            required_source_traffic(
                source_load=loads[source],
                source_capacity=capacities[source],
                perturbation=LOAD_PERTURBATION,
            )
            + SAFETY_MARGIN_MBPS
        )

        feasible_targets = [
            link
            for link in capacities
            if (
                link != source
                and target_headroom(
                    target_load=loads[link],
                    target_capacity=capacities[link],
                    perturbation=LOAD_PERTURBATION,
                )
                + FLOAT_TOLERANCE
                >= robust_required
            )
        ]

        if not feasible_targets:
            continue

        maximum_safe_budget = max(
            target_headroom(
                target_load=loads[target],
                target_capacity=capacities[target],
                perturbation=LOAD_PERTURBATION,
            )
            for target in feasible_targets
        )

        upper_budget = min(
            maximum_safe_budget - 0.25,
            robust_required + 5.0,
        )

        if upper_budget < robust_required:
            continue

        budget = round(
            rng.uniform(robust_required, upper_budget),
            2,
        )

        severity = min(
            1.0,
            required_source_traffic(
                loads[source],
                capacities[source],
                perturbation=0.0,
            )
            / 18.0,
        )

        return NetworkTask(
            task_id=task_id,
            task_type="routine_single_link",
            capacities=copy.deepcopy(capacities),
            loads=loads,
            traffic_budget_mbps=budget,
            max_actions=1,
            forbidden_target_links=[],
            complexity=min(0.49, 0.20 + 0.25 * severity),
            feasible_sources=[source],
            feasible_targets=feasible_targets,
            robust_required_traffic={
                source: robust_required,
            },
        )

    raise RuntimeError("Failed to generate a routine task.")


def generate_multi_link_task(
    task_id: int,
    rng: random.Random,
) -> NetworkTask:
    """
    Generate a task with two overloaded source links.

    A feasible solution exists, but at least two coordinated actions are
    required. The lightweight DT intentionally treats this as out of scope.
    """
    capacities = {
        "L1": 100.0,
        "L2": 100.0,
        "L3": 100.0,
        "L4": 100.0,
    }

    links = list(capacities.keys())

    for _ in range(MAX_TASK_GENERATION_ATTEMPTS):
        sources = rng.sample(links, 2)
        targets = [
            link
            for link in links
            if link not in sources
        ]

        loads = {
            link: round(rng.uniform(25.0, 45.0), 2)
            for link in links
        }

        for source in sources:
            loads[source] = round(
                rng.uniform(95.0, 106.0),
                2,
            )

        required = {
            source: ceil_to_two_decimals(
                required_source_traffic(
                    source_load=loads[source],
                    source_capacity=capacities[source],
                    perturbation=LOAD_PERTURBATION,
                )
                + SAFETY_MARGIN_MBPS
            )
            for source in sources
        }

        target_order = sorted(
            targets,
            key=lambda link: loads[link],
        )

        assignments = list(zip(sources, target_order))

        feasible = True

        for source, target in assignments:
            if (
                target_headroom(
                    target_load=loads[target],
                    target_capacity=capacities[target],
                    perturbation=LOAD_PERTURBATION,
                )
                + FLOAT_TOLERANCE
                < required[source]
            ):
                feasible = False
                break

        if not feasible:
            continue

        total_required = sum(required.values())

        budget = round(
            total_required + rng.uniform(0.5, 3.0),
            2,
        )

        return NetworkTask(
            task_id=task_id,
            task_type="complex_multi_link",
            capacities=copy.deepcopy(capacities),
            loads=loads,
            traffic_budget_mbps=budget,
            max_actions=2,
            forbidden_target_links=[],
            complexity=round(rng.uniform(0.70, 0.90), 3),
            feasible_sources=sources,
            feasible_targets=target_order,
            robust_required_traffic=required,
        )

    raise RuntimeError("Failed to generate a multi-link task.")


def generate_policy_task(
    task_id: int,
    rng: random.Random,
) -> NetworkTask:
    """
    Generate a single-link congestion task with a semantic target constraint.

    The least-loaded target is marked as forbidden, while another robustly
    feasible target remains available. A general reasoning system can use the
    policy field, but the lightweight DT baseline declares semantic-policy
    tasks outside its local rule scope.
    """
    capacities = {
        "L1": 100.0,
        "L2": 100.0,
        "L3": 100.0,
        "L4": 100.0,
    }

    links = list(capacities.keys())

    for _ in range(MAX_TASK_GENERATION_ATTEMPTS):
        source = rng.choice(links)

        loads = {
            link: round(rng.uniform(25.0, 58.0), 2)
            for link in links
        }

        loads[source] = round(
            rng.uniform(94.0, 107.0),
            2,
        )

        robust_required = ceil_to_two_decimals(
            required_source_traffic(
                source_load=loads[source],
                source_capacity=capacities[source],
                perturbation=LOAD_PERTURBATION,
            )
            + SAFETY_MARGIN_MBPS
        )

        candidate_targets = sorted(
            [
                link
                for link in links
                if link != source
            ],
            key=lambda link: loads[link],
        )

        if len(candidate_targets) < 2:
            continue

        forbidden_target = candidate_targets[0]

        allowed_feasible_targets = [
            link
            for link in candidate_targets[1:]
            if (
                target_headroom(
                    target_load=loads[link],
                    target_capacity=capacities[link],
                    perturbation=LOAD_PERTURBATION,
                )
                + FLOAT_TOLERANCE
                >= robust_required
            )
        ]

        if not allowed_feasible_targets:
            continue

        maximum_safe_budget = max(
            target_headroom(
                target_load=loads[target],
                target_capacity=capacities[target],
                perturbation=LOAD_PERTURBATION,
            )
            for target in allowed_feasible_targets
        )

        upper_budget = min(
            maximum_safe_budget - 0.25,
            robust_required + 4.0,
        )

        if upper_budget < robust_required:
            continue

        budget = round(
            rng.uniform(robust_required, upper_budget),
            2,
        )

        return NetworkTask(
            task_id=task_id,
            task_type="complex_policy_constraint",
            capacities=copy.deepcopy(capacities),
            loads=loads,
            traffic_budget_mbps=budget,
            max_actions=1,
            forbidden_target_links=[forbidden_target],
            complexity=round(rng.uniform(0.65, 0.85), 3),
            feasible_sources=[source],
            feasible_targets=allowed_feasible_targets,
            robust_required_traffic={
                source: robust_required,
            },
        )

    raise RuntimeError("Failed to generate a policy task.")


def calculate_exact_task_counts(
    num_tasks: int,
) -> Tuple[int, int, int]:
    """
    Convert the requested workload ratios into deterministic task counts.

    Examples:
        100 tasks -> 60 routine / 20 multi-link / 20 policy
         20 tasks -> 12 routine /  4 multi-link /  4 policy
    """
    if num_tasks <= 0:
        raise ValueError("NUM_TASKS must be positive.")

    routine_count = int(round(num_tasks * ROUTINE_TASK_RATIO))
    multi_count = int(round(num_tasks * MULTI_LINK_TASK_RATIO))
    policy_count = num_tasks - routine_count - multi_count

    if policy_count < 0:
        raise ValueError("Invalid task-ratio configuration.")

    return routine_count, multi_count, policy_count


def generate_tasks(
    num_tasks: int,
    seed: int,
) -> List[NetworkTask]:
    """
    Generate an exact 60/20/20 workload (for NUM_TASKS=100), then shuffle it.

    task_type is retained only as an evaluation label. The DT controller does
    not use it to decide whether a task is within its local capability.
    """
    rng = random.Random(seed)

    routine_count, multi_count, policy_count = calculate_exact_task_counts(
        num_tasks
    )

    tasks = []
    temporary_id = 0

    for _ in range(routine_count):
        tasks.append(
            generate_routine_task(
                task_id=temporary_id,
                rng=rng,
            )
        )
        temporary_id += 1

    for _ in range(multi_count):
        tasks.append(
            generate_multi_link_task(
                task_id=temporary_id,
                rng=rng,
            )
        )
        temporary_id += 1

    for _ in range(policy_count):
        tasks.append(
            generate_policy_task(
                task_id=temporary_id,
                rng=rng,
            )
        )
        temporary_id += 1

    rng.shuffle(tasks)

    # Reassign IDs after shuffling so IDs match the execution order.
    for new_task_id, task in enumerate(tasks):
        task.task_id = new_task_id

    return tasks

# ============================================================
# 5. Shared plan evaluator / DT sandbox
# ============================================================

class PlanEvaluator:
    """
    Common evaluator for future DT-only, LAM-only, and Octopus scripts.
    """

    @staticmethod
    def validate_plan(
        task: NetworkTask,
        plan: Optional[CandidatePlan],
    ) -> Tuple[bool, str]:
        if plan is None:
            return False, "No candidate plan was generated."

        if not plan.actions:
            return False, "The candidate plan contains no action."

        if len(plan.actions) > task.max_actions:
            return False, (
                "The plan contains {} actions, exceeding max_actions={}."
            ).format(
                len(plan.actions),
                task.max_actions,
            )

        total_moved = sum(
            action.traffic
            for action in plan.actions
        )

        if (
            total_moved
            > task.traffic_budget_mbps + FLOAT_TOLERANCE
        ):
            return False, (
                "Total moved traffic {:.6f} Mbps exceeds budget "
                "{:.6f} Mbps."
            ).format(
                total_moved,
                task.traffic_budget_mbps,
            )

        simulated_loads = copy.deepcopy(task.loads)

        for action_index, action in enumerate(
            plan.actions,
            start=1,
        ):
            if action.action != "reroute":
                return False, (
                    "Action {} has an unsupported type."
                ).format(action_index)

            if action.from_link not in simulated_loads:
                return False, (
                    "Action {} uses an unknown source link."
                ).format(action_index)

            if action.to_link not in simulated_loads:
                return False, (
                    "Action {} uses an unknown target link."
                ).format(action_index)

            if action.from_link == action.to_link:
                return False, (
                    "Action {} uses the same source and target."
                ).format(action_index)

            if action.to_link in task.forbidden_target_links:
                return False, (
                    "Action {} violates the forbidden-target policy."
                ).format(action_index)

            if action.traffic <= 0:
                return False, (
                    "Action {} has non-positive traffic."
                ).format(action_index)

            if (
                action.traffic
                > simulated_loads[action.from_link]
                + FLOAT_TOLERANCE
            ):
                return False, (
                    "Action {} moves more traffic than the source load."
                ).format(action_index)

            simulated_loads[action.from_link] -= action.traffic
            simulated_loads[action.to_link] += action.traffic

        return True, "The candidate plan is structurally valid."

    @staticmethod
    def apply_plan(
        loads: Dict[str, float],
        plan: CandidatePlan,
    ) -> Dict[str, float]:
        updated = copy.deepcopy(loads)

        for action in plan.actions:
            updated[action.from_link] -= action.traffic
            updated[action.to_link] += action.traffic

        return updated

    def evaluate(
        self,
        task: NetworkTask,
        plan: Optional[CandidatePlan],
        scenarios: int = SANDBOX_SCENARIOS,
    ) -> EffectMetrics:
        before_max, before_overloaded = state_metrics(
            task.loads,
            task.capacities,
        )

        plan_generated = (
            plan is not None
            and len(plan.actions) > 0
        )

        structurally_valid, validation_message = (
            self.validate_plan(task, plan)
        )

        if not structurally_valid or plan is None:
            return EffectMetrics(
                plan_generated=plan_generated,
                structurally_valid=False,
                nominal_passed=False,
                robust_passed=False,

                before_max_utilization=before_max,
                after_max_utilization=before_max,
                max_utilization_reduction=0.0,

                overloaded_links_before=before_overloaded,
                overloaded_links_after=before_overloaded,
                overloaded_link_reduction=0,

                sandbox_pass_ratio=0.0,
                feedback=validation_message,
            )

        nominal_after = self.apply_plan(
            task.loads,
            plan,
        )

        after_max, after_overloaded = state_metrics(
            nominal_after,
            task.capacities,
        )

        nominal_passed = (
            after_overloaded == 0
            and after_max
            <= OVERLOAD_THRESHOLD + FLOAT_TOLERANCE
        )

        rng = random.Random(task.task_id + 10000)
        passed_scenarios = 0

        for _ in range(scenarios):
            perturbed_loads = {
                link: max(
                    0.0,
                    load * rng.uniform(
                        1.0 - LOAD_PERTURBATION,
                        1.0 + LOAD_PERTURBATION,
                    ),
                )
                for link, load in task.loads.items()
            }

            perturbed_after = self.apply_plan(
                perturbed_loads,
                plan,
            )

            perturbed_max, perturbed_overloaded = state_metrics(
                perturbed_after,
                task.capacities,
            )

            scenario_passed = (
                perturbed_overloaded == 0
                and perturbed_max
                <= OVERLOAD_THRESHOLD + FLOAT_TOLERANCE
            )

            if scenario_passed:
                passed_scenarios += 1

        sandbox_pass_ratio = passed_scenarios / scenarios

        robust_passed = (
            sandbox_pass_ratio + FLOAT_TOLERANCE
            >= ROBUST_PASS_RATIO
        )

        feedback = (
            "Nominal pass={}; sandbox pass ratio={:.3f}; "
            "maximum utilization={:.3f}->{:.3f}; "
            "overloaded links={}->{}"
        ).format(
            nominal_passed,
            sandbox_pass_ratio,
            before_max,
            after_max,
            before_overloaded,
            after_overloaded,
        )

        return EffectMetrics(
            plan_generated=True,
            structurally_valid=True,
            nominal_passed=nominal_passed,
            robust_passed=robust_passed,

            before_max_utilization=before_max,
            after_max_utilization=after_max,
            max_utilization_reduction=(
                before_max - after_max
            ),

            overloaded_links_before=before_overloaded,
            overloaded_links_after=after_overloaded,
            overloaded_link_reduction=(
                before_overloaded - after_overloaded
            ),

            sandbox_pass_ratio=sandbox_pass_ratio,
            feedback=feedback,
        )

# ============================================================
# 6. Prompt construction and output parsing
# ============================================================

def build_prompt(task: NetworkTask) -> str:
    # Deliberately exclude task.task_type and diagnostic feasibility fields.
    # The LAM must infer the scenario from observable state and constraints.
    network_state = {
        "capacities_mbps": task.capacities,
        "reported_loads_mbps": task.loads,
        "total_traffic_budget_mbps": task.traffic_budget_mbps,
        "maximum_number_of_actions": task.max_actions,
        "forbidden_target_links": task.forbidden_target_links,
        "maximum_allowed_link_utilization": OVERLOAD_THRESHOLD,
        "load_uncertainty_fraction": LOAD_PERTURBATION,
        "load_uncertainty_description": (
            "Before execution, every link load may independently vary "
            "between 95% and 105% of its reported value."
        ),
        "required_sandbox_pass_ratio": ROBUST_PASS_RATIO,
        "recommended_safety_margin_mbps": SAFETY_MARGIN_MBPS,
    }

    return """
You are a network-management model.

Inspect the network state and generate one robust traffic-management plan.
You must infer the required number of rerouting actions from the reported
loads and the provided constraints; no task-category label is given.

Network state:
{network_state}

Uncertainty model:
1. Reported link loads are not perfectly stable.
2. Before execution, every link load may independently vary within +/-5%.
3. Account for the worst-case +5% load of every source and target link.
4. Leave approximately 0.50 Mbps of safety margin below the overload boundary
   when the traffic budget permits.

Hard plan constraints:
1. Use only links listed in capacities_mbps.
2. Every action must have different source and target links.
3. Identify all links whose reported utilization exceeds the allowed limit and
   relieve every such overloaded source.
4. If multiple links are overloaded, coordinate multiple rerouting actions when
   permitted by maximum_number_of_actions.
5. Never use a link in forbidden_target_links as a target.
6. The number of actions must not exceed maximum_number_of_actions.
7. The sum of moved traffic across all actions must satisfy:
   0 < total moved traffic <= total_traffic_budget_mbps.
8. After all actions, every link utilization should be no greater than
   maximum_allowed_link_utilization under the stated uncertainty.
9. Do not create a new overloaded target link.
10. First satisfy the robust constraints; only then minimize total moved traffic.
11. The plan should pass at least 95% of the DT sandbox scenarios.
12. Return JSON only. Do not return Markdown, calculations, or explanations.

Required JSON format:
{{
  "actions": [
    {{
      "action": "reroute",
      "from_link": "<source link>",
      "to_link": "<target link>",
      "traffic": <numeric value>
    }}
  ]
}}
""".strip().format(
        network_state=json.dumps(
            network_state,
            ensure_ascii=False,
            indent=2,
        )
    )


def extract_json(text: str) -> Dict[str, Any]:
    cleaned_text = text.strip()

    cleaned_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    json_match = re.search(
        r"\{.*\}",
        cleaned_text,
        flags=re.DOTALL,
    )

    if not json_match:
        raise ValueError(
            "No JSON object was found in the LAM output."
        )

    return json.loads(json_match.group(0))


def parse_plan(text: str) -> CandidatePlan:
    data = extract_json(text)

    # Accept the required multi-action format.
    if "actions" in data:
        raw_actions = data["actions"]

    # Also accept one top-level action and wrap it as a one-action plan.
    elif {
        "action",
        "from_link",
        "to_link",
        "traffic",
    }.issubset(set(data.keys())):
        raw_actions = [data]

    else:
        raise ValueError(
            "The output must contain an 'actions' list."
        )

    if not isinstance(raw_actions, list):
        raise ValueError("'actions' must be a list.")

    if len(raw_actions) == 0:
        raise ValueError("'actions' must not be empty.")

    actions = []

    for action_index, raw_action in enumerate(
        raw_actions,
        start=1,
    ):
        if not isinstance(raw_action, dict):
            raise ValueError(
                "Action {} is not a JSON object.".format(
                    action_index
                )
            )

        required_fields = {
            "action",
            "from_link",
            "to_link",
            "traffic",
        }

        missing_fields = (
            required_fields - set(raw_action.keys())
        )

        if missing_fields:
            raise ValueError(
                "Action {} is missing fields: {}".format(
                    action_index,
                    sorted(missing_fields),
                )
            )

        actions.append(
            CandidateAction(
                action=str(
                    raw_action["action"]
                ).strip(),
                from_link=str(
                    raw_action["from_link"]
                ).strip(),
                to_link=str(
                    raw_action["to_link"]
                ).strip(),
                traffic=float(
                    raw_action["traffic"]
                ),
            )
        )

    return CandidatePlan(actions=actions)

# ============================================================
# 7. Real LAM API call
# ============================================================

def call_lam_once(task: NetworkTask) -> LAMMeasurement:
    prompt = build_prompt(task)

    if PRINT_PROMPT:
        print("\n========== Prompt sent to LAM ==========")
        print(prompt)
        print("========================================\n")

    start_time = time.perf_counter()
    first_token_time = None
    output_parts = []

    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate constrained, structured, robust, "
                        "and policy-compliant network-management plans."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            stream=True,

            # 关闭 DeepSeek Thinking Mode
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if not content:
                continue

            if first_token_time is None:
                first_token_time = time.perf_counter()

            output_parts.append(content)

        end_time = time.perf_counter()
        output_text = "".join(output_parts).strip()

        if first_token_time is None:
            ttft_ms = (
                end_time - start_time
            ) * 1000.0
        else:
            ttft_ms = (
                first_token_time - start_time
            ) * 1000.0

        total_latency_ms = (
            end_time - start_time
        ) * 1000.0

        try:
            plan = parse_plan(output_text)
            parse_success = True
            error_message = ""
        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as parse_error:
            plan = None
            parse_success = False
            error_message = (
                "Parse error: {}".format(parse_error)
            )

        return LAMMeasurement(
            plan=plan,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_text=output_text,
            api_success=True,
            parse_success=parse_success,
            error_message=error_message,
        )

    except Exception as api_error:
        end_time = time.perf_counter()

        return LAMMeasurement(
            plan=None,
            ttft_ms=0.0,
            total_latency_ms=(
                end_time - start_time
            ) * 1000.0,
            output_text="",
            api_success=False,
            parse_success=False,
            error_message=(
                "API error: {}".format(api_error)
            ),
        )


def call_lam(task: NetworkTask) -> LAMMeasurement:
    last_measurement = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        measurement = call_lam_once(task)
        last_measurement = measurement

        if measurement.api_success:
            return measurement

        print(
            "  API attempt {}/{} failed: {}".format(
                attempt,
                MAX_API_RETRIES,
                measurement.error_message,
            )
        )

        if attempt < MAX_API_RETRIES:
            time.sleep(RETRY_INTERVAL_SECONDS)

    if last_measurement is None:
        raise RuntimeError(
            "LAM request failed without a returned measurement."
        )

    return last_measurement

# ============================================================
# 8. Result construction and reporting
# ============================================================

def make_result(
    task: NetworkTask,
    measurement: LAMMeasurement,
    effect: EffectMetrics,
    verification_latency_ms: float,
    service_latency_ms: float,
) -> ExperimentResult:
    plan = measurement.plan

    action_count = (
        len(plan.actions)
        if plan is not None
        else 0
    )

    total_moved = (
        sum(action.traffic for action in plan.actions)
        if plan is not None
        else 0.0
    )

    return ExperimentResult(
        architecture="LAM-only",
        task_id=task.task_id,
        task_type=task.task_type,
        task_complexity=task.complexity,

        lam_inference_latency_ms=measurement.total_latency_ms,
        verification_latency_ms=verification_latency_ms,
        service_latency_ms=service_latency_ms,
        ttft_ms=measurement.ttft_ms,

        api_success=int(measurement.api_success),
        parse_success=int(measurement.parse_success),

        strategy_generated=int(effect.plan_generated),
        unresolved_task=int(not effect.plan_generated),
        invalid_generated_strategy=int(
            effect.plan_generated
            and not effect.structurally_valid
        ),

        nominal_success=int(
            effect.structurally_valid
            and effect.nominal_passed
        ),
        robust_success=int(
            effect.structurally_valid
            and effect.robust_passed
        ),

        before_max_utilization=effect.before_max_utilization,
        after_max_utilization=effect.after_max_utilization,
        max_utilization_reduction=effect.max_utilization_reduction,

        overloaded_links_before=effect.overloaded_links_before,
        overloaded_links_after=effect.overloaded_links_after,
        overloaded_link_reduction=effect.overloaded_link_reduction,

        sandbox_pass_ratio=effect.sandbox_pass_ratio,

        action_count=action_count,
        total_moved_traffic_mbps=total_moved,
        traffic_budget_mbps=task.traffic_budget_mbps,

        lam_invoked=1,
        lam_calls=1,

        action_plan=plan_to_string(plan),
        raw_lam_output=measurement.output_text,
        error_message=measurement.error_message,
    )


def percentile_nearest_rank(
    values: List[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = max(
        0,
        math.ceil(percentile * len(sorted_values)) - 1,
    )
    return sorted_values[index]


def rate(
    rows: List[ExperimentResult],
    field_name: str,
) -> float:
    if not rows:
        return 0.0

    return 100.0 * mean(
        getattr(row, field_name)
        for row in rows
    )


def summarize(
    results: List[ExperimentResult],
) -> Dict[str, Any]:
    """
    Report only the five architecture-level metrics used in the main table.

    1. Average Latency:
       Average end-to-end service latency over all tasks.

    2. Task Success Rate:
       Fraction of all tasks that are structurally valid and successfully
       solve the nominal network state.

    3. Successful-Strategy Robustness:
       Average sandbox pass ratio calculated ONLY over nominally successful
       strategies. Failed or unresolved tasks are excluded.

    4. Avg. Max-Util. Reduction:
       Average reduction of the maximum link utilization over all tasks.

    5. LAM Invocation Ratio:
       Fraction of tasks that invoke the LAM.
    """
    if not results:
        raise ValueError("No experiment results are available.")

    successful_rows = [
        row
        for row in results
        if row.nominal_success == 1
    ]

    average_latency_ms = mean(
        row.service_latency_ms
        for row in results
    )

    task_success_rate = 100.0 * mean(
        row.nominal_success
        for row in results
    )

    successful_strategy_robustness = (
        100.0 * mean(
            row.sandbox_pass_ratio
            for row in successful_rows
        )
        if successful_rows
        else 0.0
    )

    avg_max_util_reduction_pp = 100.0 * mean(
        row.max_utilization_reduction
        for row in results
    )

    lam_invocation_ratio = 100.0 * mean(
        row.lam_invoked
        for row in results
    )

    return {
        "Architecture": "LAM-only",
        "Average Latency (ms)": round(
            average_latency_ms,
            2,
        ),
        "Task Success Rate (%)": round(
            task_success_rate,
            2,
        ),
        "Successful-Strategy Robustness (%)": round(
            successful_strategy_robustness,
            2,
        ),
        "Avg. Max-Util. Reduction (pp)": round(
            avg_max_util_reduction_pp,
            2,
        ),
        "LAM Invocation Ratio (%)": round(
            lam_invocation_ratio,
            2,
        ),
    }

def save_csv(
    path: str,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError("No rows to save.")

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    summary: Dict[str, Any],
) -> None:
    print("\n========== LAM-only Summary ==========")
    for key, value in summary.items():
        print("{}: {}".format(key, value))


# ============================================================
# 9. Main experiment
# ============================================================

def main() -> None:
    evaluator = PlanEvaluator()

    tasks = generate_tasks(
        num_tasks=NUM_TASKS,
        seed=RANDOM_SEED,
    )

    routine_count = sum(
        task.task_type == "routine_single_link"
        for task in tasks
    )
    multi_count = sum(
        task.task_type == "complex_multi_link"
        for task in tasks
    )
    policy_count = sum(
        task.task_type == "complex_policy_constraint"
        for task in tasks
    )

    print(
        "Generated workload: {} routine, {} multi-link complex, "
        "{} policy-constrained complex.".format(
            routine_count,
            multi_count,
            policy_count,
        )
    )

    results = []

    for task_index, task in enumerate(tasks, start=1):
        print(
            "\nRunning LAM-only task {}/{}".format(
                task_index,
                len(tasks),
            )
        )

        # This label is printed only for experiment logging; it is not sent
        # to the LAM in build_prompt().
        print("  Task label: {}".format(task.task_type))
        print("  Loads: {}".format(task.loads))
        print(
            "  Traffic budget: {:.2f} Mbps".format(
                task.traffic_budget_mbps
            )
        )
        print("  Maximum actions: {}".format(task.max_actions))
        print("  Forbidden targets: {}".format(task.forbidden_target_links))

        service_start = time.perf_counter()

        measurement = call_lam(task)

        if measurement.api_success:
            print(
                "  LAM inference latency: {:.2f} ms".format(
                    measurement.total_latency_ms
                )
            )
            print("  TTFT: {:.2f} ms".format(measurement.ttft_ms))
            print("  Raw output: {}".format(measurement.output_text))
        else:
            print(
                "  API request failed: {}".format(
                    measurement.error_message
                )
            )

        verification_start = time.perf_counter()

        effect = evaluator.evaluate(
            task=task,
            plan=measurement.plan,
        )

        verification_end = time.perf_counter()
        service_end = verification_end

        verification_latency_ms = (
            verification_end - verification_start
        ) * 1000.0

        service_latency_ms = (
            service_end - service_start
        ) * 1000.0

        print("  Parsed plan: {}".format(plan_to_string(measurement.plan)))
        print("  Evaluation: {}".format(effect.feedback))
        print(
            "  Verification latency: {:.3f} ms".format(
                verification_latency_ms
            )
        )
        print(
            "  Total service latency: {:.2f} ms".format(
                service_latency_ms
            )
        )

        results.append(
            make_result(
                task=task,
                measurement=measurement,
                effect=effect,
                verification_latency_ms=verification_latency_ms,
                service_latency_ms=service_latency_ms,
            )
        )

    summary = summarize(results)
    print_summary(summary)

    save_csv(
        "lam_only_mixed_detailed_results.csv",
        [asdict(result) for result in results],
    )

    save_csv(
        "lam_only_mixed_summary.csv",
        [summary],
    )

    print("\nSaved:")
    print("  lam_only_mixed_detailed_results.csv")
    print("  lam_only_mixed_summary.csv")


if __name__ == "__main__":
    main()


