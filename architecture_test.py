
"""
Unified comparison of three architectures:
1. DT-only
2. LAM-only
3. Octopus (OCP)

Fair-comparison design
----------------------
- All three architectures use exactly the SAME 20 network-management requests.
- Exact workload composition:
    12 Routine
     4 Multi-link
     4 Policy-constrained
- Four links, each with 100 Mbps capacity.
- Overload threshold: 90%.
- Each candidate strategy is evaluated under 1000 independent +/-5% load
  perturbation scenarios.
- DeepSeek V4 Flash is used with Thinking Mode disabled.
- LAM-only invokes the LAM exactly once for every request.
- Octopus first tries the local DT; if it cannot solve the request or its
  candidate fails robust verification, the request is escalated to the LAM.
- Each Octopus request invokes the LAM at most twice. If round 1 fails,
  round 2 receives the COMPLETE round-1 prompt, the COMPLETE round-1 raw
  output, and the detailed DT verification/rejection feedback.
- Octopus prints and saves the full Prompt -> Output -> DT Verification
  trace for every LAM round.
- LAM overhead is summarized using the total number of actual LLM/API
  accesses. Every execution of client.chat.completions.create(...) counts
  once, so a two-round Octopus request contributes 2 calls, and API retries
  are counted as additional accesses.
- The artificial task-type label is NEVER given to the local DT or the LAM.
- Success is defined identically for all architectures as safe deployability:
  the candidate must be structurally valid and achieve at least the configured
  DT sandbox pass threshold.
- Rejected candidates are not counted as successful and contribute zero
  physical-network improvement.
- The LAM is explicitly informed that reported loads may fluctuate by +/-5%
  before execution and must account for this uncertainty.

Outputs
-------
three_architecture_detailed_results.csv
three_architecture_summary.csv
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
# 1. LAM configuration
# ============================================================

client = OpenAI(
    api_key="sk-73e3fc16c9104b0ab5d26a809f313d5f",
    base_url="https://api.deepseek.com"
)

MODEL_NAME = "deepseek-v4-pro"


# ============================================================
# 2. Experiment configuration
# ============================================================

NUM_TASKS = 100
RANDOM_SEED = 2026

ROUTINE_TASK_RATIO = 0.60
MULTI_LINK_TASK_RATIO = 0.20
POLICY_TASK_RATIO = 0.20

LINKS = ["L1", "L2", "L3", "L4"]
LINK_CAPACITY_MBPS = 100.0

OVERLOAD_THRESHOLD = 0.90
LOAD_PERTURBATION = 0.05
SANDBOX_SCENARIOS = 1000

# Octopus deployment threshold.
ROBUST_PASS_RATIO = 0.95

SAFETY_MARGIN_MBPS = 0.50
FLOAT_TOLERANCE = 1e-6

MAX_LAM_CALLS_PER_OCTOPUS_REQUEST = 2

MAX_API_RETRIES = 3
RETRY_INTERVAL_SECONDS = 2.0

# Turn on only when you want to inspect every full LAM prompt.
PRINT_PROMPT = False


# ============================================================
# 3. Data structures
# ============================================================

@dataclass
class NetworkTask:
    task_id: int

    # Used only to group results after the experiment.
    # Neither DT nor LAM receives this artificial label.
    task_type: str

    capacities: Dict[str, float]
    loads: Dict[str, float]

    traffic_budget_mbps: float
    max_actions: int

    # Hidden machine-readable policy used by the evaluator.
    forbidden_target_links: List[str]

    # Natural-language policy shown to the LAM.
    policy_text: str


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

    # Full prompt actually sent to the LAM.
    prompt_text: str

    # Raw model output.
    output_text: str

    api_success: bool
    parse_success: bool
    error_message: str

    # Number of actual LLM/API accesses consumed by this logical call.
    # Each execution of client.chat.completions.create(...) counts as 1.
    api_call_count: int = 1


@dataclass
class EvaluationResult:
    plan_generated: bool
    structurally_valid: bool
    nominal_success: bool

    sandbox_pass_ratio: float
    robust_success: bool

    before_max_utilization: float
    after_max_utilization: float
    max_utilization_reduction: float

    feedback: str


@dataclass
class ArchitectureResult:
    architecture: str
    task_id: int
    task_type: str

    processing_path: str

    service_latency_ms: float
    ttft_ms: float

    task_success: int
    deployed: int

    sandbox_pass_ratio: float
    max_utilization_reduction_pp: float

    lam_invoked: int
    lam_calls: int

    action_count: int
    total_moved_traffic_mbps: float

    final_plan: str
    feedback: str
    raw_lam_output: str
    error_message: str


# ============================================================
# 4. Common utility functions
# ============================================================

def utilization(
    load: float,
    capacity: float,
) -> float:
    if capacity <= 0:
        raise ValueError("Capacity must be positive.")
    return load / capacity


def state_metrics(
    loads: Dict[str, float],
    capacities: Dict[str, float],
) -> Tuple[float, int]:
    utils = {
        link: utilization(loads[link], capacities[link])
        for link in loads
    }

    max_util = max(utils.values())

    overloaded = sum(
        value > OVERLOAD_THRESHOLD + FLOAT_TOLERANCE
        for value in utils.values()
    )

    return max_util, overloaded


def find_overloaded_links(
    task: NetworkTask,
) -> List[str]:
    return [
        link
        for link, load in task.loads.items()
        if (
            utilization(load, task.capacities[link])
            > OVERLOAD_THRESHOLD + FLOAT_TOLERANCE
        )
    ]


def required_source_traffic(
    source_load: float,
    source_capacity: float,
    perturbation: float = 0.0,
) -> float:
    worst_source_load = (
        source_load * (1.0 + perturbation)
    )

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
    worst_target_load = (
        target_load * (1.0 + perturbation)
    )

    return max(
        0.0,
        OVERLOAD_THRESHOLD * target_capacity
        - worst_target_load,
    )


def ceil_to_two_decimals(
    value: float,
) -> float:
    return math.ceil(
        (value - FLOAT_TOLERANCE) * 100.0
    ) / 100.0


def plan_to_string(
    plan: Optional[CandidatePlan],
) -> str:
    if plan is None:
        return "None"

    return json.dumps(
        {
            "actions": [
                asdict(action)
                for action in plan.actions
            ]
        },
        ensure_ascii=False,
    )


def base_capacities() -> Dict[str, float]:
    return {
        link: LINK_CAPACITY_MBPS
        for link in LINKS
    }


# ============================================================
# 5. Common workload generation
# ============================================================

def calculate_exact_task_counts(
    num_tasks: int,
) -> Tuple[int, int, int]:
    routine = int(
        round(num_tasks * ROUTINE_TASK_RATIO)
    )

    multi = int(
        round(num_tasks * MULTI_LINK_TASK_RATIO)
    )

    policy = (
        num_tasks - routine - multi
    )

    return routine, multi, policy


def generate_routine_task(
    task_id: int,
    rng: random.Random,
) -> NetworkTask:
    """
    One overloaded link. A robust one-action solution exists.
    """
    capacities = base_capacities()

    while True:
        source = rng.choice(LINKS)

        loads = {
            link: round(
                rng.uniform(28.0, 60.0),
                2,
            )
            for link in LINKS
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
            for link in LINKS
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

        max_safe = max(
            target_headroom(
                target_load=loads[target],
                target_capacity=capacities[target],
                perturbation=LOAD_PERTURBATION,
            )
            for target in feasible_targets
        )

        upper_budget = min(
            max_safe - 0.25,
            robust_required + 5.0,
        )

        if upper_budget < robust_required:
            continue

        budget = round(
            rng.uniform(
                robust_required,
                upper_budget,
            ),
            2,
        )

        return NetworkTask(
            task_id=task_id,
            task_type="Routine",
            capacities=capacities,
            loads=loads,
            traffic_budget_mbps=budget,
            max_actions=1,
            forbidden_target_links=[],
            policy_text=(
                "No additional routing policy is imposed."
            ),
        )


def generate_multi_link_task(
    task_id: int,
    rng: random.Random,
) -> NetworkTask:
    """
    Two links are overloaded. A feasible coordinated two-action solution exists.
    """
    capacities = base_capacities()

    while True:
        sources = rng.sample(
            LINKS,
            2,
        )

        targets = [
            link
            for link in LINKS
            if link not in sources
        ]

        loads = {
            link: round(
                rng.uniform(25.0, 45.0),
                2,
            )
            for link in LINKS
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

        feasible = True

        for source, target in zip(
            sources,
            target_order,
        ):
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

        total_required = sum(
            required.values()
        )

        budget = round(
            total_required
            + rng.uniform(0.5, 3.0),
            2,
        )

        return NetworkTask(
            task_id=task_id,
            task_type="Multi-link",
            capacities=capacities,
            loads=loads,
            traffic_budget_mbps=budget,
            max_actions=2,
            forbidden_target_links=[],
            policy_text=(
                "No additional routing policy is imposed."
            ),
        )


def generate_policy_task(
    task_id: int,
    rng: random.Random,
) -> NetworkTask:
    """
    One overloaded link. The least-loaded target is placed under maintenance.
    Another robustly feasible target remains available.
    """
    capacities = base_capacities()

    while True:
        source = rng.choice(LINKS)

        loads = {
            link: round(
                rng.uniform(25.0, 58.0),
                2,
            )
            for link in LINKS
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
                for link in LINKS
                if link != source
            ],
            key=lambda link: loads[link],
        )

        forbidden_target = (
            candidate_targets[0]
        )

        allowed_targets = [
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

        if not allowed_targets:
            continue

        max_safe = max(
            target_headroom(
                target_load=loads[target],
                target_capacity=capacities[target],
                perturbation=LOAD_PERTURBATION,
            )
            for target in allowed_targets
        )

        upper_budget = min(
            max_safe - 0.25,
            robust_required + 4.0,
        )

        if upper_budget < robust_required:
            continue

        budget = round(
            rng.uniform(
                robust_required,
                upper_budget,
            ),
            2,
        )

        return NetworkTask(
            task_id=task_id,
            task_type="Policy",
            capacities=capacities,
            loads=loads,
            traffic_budget_mbps=budget,
            max_actions=1,
            forbidden_target_links=[
                forbidden_target
            ],
            policy_text=(
                f"Link {forbidden_target} is under temporary maintenance "
                f"and must not receive any rerouted traffic."
            ),
        )


def generate_common_tasks(
    num_tasks: int,
    seed: int,
) -> List[NetworkTask]:
    """
    Create one common task list shared by ALL architectures.
    """
    rng = random.Random(seed)

    routine_count, multi_count, policy_count = (
        calculate_exact_task_counts(num_tasks)
    )

    tasks = []
    temporary_id = 0

    for _ in range(routine_count):
        tasks.append(
            generate_routine_task(
                temporary_id,
                rng,
            )
        )
        temporary_id += 1

    for _ in range(multi_count):
        tasks.append(
            generate_multi_link_task(
                temporary_id,
                rng,
            )
        )
        temporary_id += 1

    for _ in range(policy_count):
        tasks.append(
            generate_policy_task(
                temporary_id,
                rng,
            )
        )
        temporary_id += 1

    rng.shuffle(tasks)

    # Reassign IDs after shuffling.
    # The same IDs are then used by all architectures, so sandbox
    # perturbation sequences are also identical.
    for new_id, task in enumerate(tasks):
        task.task_id = new_id

    return tasks


# ============================================================
# 6. Lightweight local DT
# ============================================================

class DigitalTwin:
    """
    A deterministic lightweight local rule.

    It does NOT read task.task_type.

    Capability boundary:
    - supports exactly one overloaded link;
    - supports one rerouting action;
    - does not interpret non-trivial semantic policy text.
    """

    def local_analysis(
        self,
        task: NetworkTask,
    ) -> Tuple[
        Optional[CandidatePlan],
        str,
    ]:
        overloaded_links = (
            find_overloaded_links(task)
        )

        if len(overloaded_links) != 1:
            return (
                None,
                (
                    "Out of local DT scope: the local rule "
                    "supports exactly one overloaded link."
                ),
            )

        # Semantic policy handling is intentionally outside this lightweight DT.
        if (
            task.policy_text
            != "No additional routing policy is imposed."
        ):
            return (
                None,
                (
                    "Out of local DT scope: the request contains "
                    "a semantic routing policy."
                ),
            )

        if task.max_actions != 1:
            return (
                None,
                (
                    "Out of local DT scope: coordinated "
                    "multi-action control is required."
                ),
            )

        source = overloaded_links[0]

        robust_traffic = ceil_to_two_decimals(
            required_source_traffic(
                source_load=task.loads[source],
                source_capacity=task.capacities[source],
                perturbation=LOAD_PERTURBATION,
            )
            + SAFETY_MARGIN_MBPS
        )

        if (
            robust_traffic
            > task.traffic_budget_mbps
            + FLOAT_TOLERANCE
        ):
            return (
                None,
                (
                    "Required robust migration exceeds "
                    "the traffic budget."
                ),
            )

        target_candidates = sorted(
            [
                link
                for link in task.loads
                if link != source
            ],
            key=lambda link: utilization(
                task.loads[link],
                task.capacities[link],
            ),
        )

        for target in target_candidates:
            if (
                target_headroom(
                    target_load=task.loads[target],
                    target_capacity=task.capacities[target],
                    perturbation=LOAD_PERTURBATION,
                )
                + FLOAT_TOLERANCE
                >= robust_traffic
            ):
                return (
                    CandidatePlan(
                        actions=[
                            CandidateAction(
                                action="reroute",
                                from_link=source,
                                to_link=target,
                                traffic=robust_traffic,
                            )
                        ]
                    ),
                    (
                        f"Local DT moved {robust_traffic:.2f} Mbps "
                        f"from {source} to {target}."
                    ),
                )

        return (
            None,
            "No robust target is available.",
        )


# ============================================================
# 7. Shared sandbox evaluator
# ============================================================

class PlanEvaluator:

    @staticmethod
    def validate_plan(
        task: NetworkTask,
        plan: Optional[CandidatePlan],
    ) -> Tuple[bool, str]:
        if plan is None:
            return False, "No candidate plan."

        if not plan.actions:
            return False, "No action in candidate plan."

        if len(plan.actions) > task.max_actions:
            return (
                False,
                "Too many actions.",
            )

        total_moved = sum(
            action.traffic
            for action in plan.actions
        )

        if (
            total_moved
            > task.traffic_budget_mbps
            + FLOAT_TOLERANCE
        ):
            return (
                False,
                "Traffic budget exceeded.",
            )

        simulated = copy.deepcopy(
            task.loads
        )

        for action in plan.actions:
            if action.action != "reroute":
                return False, "Unsupported action."

            if (
                action.from_link not in simulated
                or action.to_link not in simulated
            ):
                return False, "Unknown link."

            if (
                action.from_link
                == action.to_link
            ):
                return (
                    False,
                    "Source equals target.",
                )

            if (
                action.to_link
                in task.forbidden_target_links
            ):
                return (
                    False,
                    "Policy constraint violated.",
                )

            if action.traffic <= 0:
                return (
                    False,
                    "Non-positive traffic.",
                )

            if (
                action.traffic
                > simulated[action.from_link]
                + FLOAT_TOLERANCE
            ):
                return (
                    False,
                    "Traffic exceeds source load.",
                )

            simulated[
                action.from_link
            ] -= action.traffic

            simulated[
                action.to_link
            ] += action.traffic

        return True, "Structurally valid."

    @staticmethod
    def apply_plan(
        loads: Dict[str, float],
        plan: CandidatePlan,
    ) -> Dict[str, float]:
        updated = copy.deepcopy(
            loads
        )

        for action in plan.actions:
            updated[
                action.from_link
            ] -= action.traffic

            updated[
                action.to_link
            ] += action.traffic

        return updated

    def evaluate(
        self,
        task: NetworkTask,
        plan: Optional[CandidatePlan],
    ) -> EvaluationResult:
        before_max, _ = state_metrics(
            task.loads,
            task.capacities,
        )

        plan_generated = (
            plan is not None
            and len(plan.actions) > 0
        )

        valid, validation_message = (
            self.validate_plan(
                task,
                plan,
            )
        )

        if (
            not valid
            or plan is None
        ):
            return EvaluationResult(
                plan_generated=plan_generated,
                structurally_valid=False,
                nominal_success=False,
                sandbox_pass_ratio=0.0,
                robust_success=False,
                before_max_utilization=before_max,
                after_max_utilization=before_max,
                max_utilization_reduction=0.0,
                feedback=validation_message,
            )

        nominal_after = self.apply_plan(
            task.loads,
            plan,
        )

        (
            after_max,
            after_overloaded,
        ) = state_metrics(
            nominal_after,
            task.capacities,
        )

        nominal_success = (
            after_overloaded == 0
            and after_max
            <= OVERLOAD_THRESHOLD
            + FLOAT_TOLERANCE
        )

        # IMPORTANT:
        # Same task ID -> same perturbation sequence for every architecture.
        rng = random.Random(
            RANDOM_SEED
            + 10000
            + task.task_id
        )

        passed = 0

        for _ in range(
            SANDBOX_SCENARIOS
        ):
            perturbed_loads = {
                link: max(
                    0.0,
                    load * rng.uniform(
                        1.0 - LOAD_PERTURBATION,
                        1.0 + LOAD_PERTURBATION,
                    ),
                )
                for link, load
                in task.loads.items()
            }

            after = self.apply_plan(
                perturbed_loads,
                plan,
            )

            (
                scenario_max,
                scenario_overloaded,
            ) = state_metrics(
                after,
                task.capacities,
            )

            if (
                scenario_overloaded == 0
                and scenario_max
                <= OVERLOAD_THRESHOLD
                + FLOAT_TOLERANCE
            ):
                passed += 1

        sandbox_ratio = (
            passed / SANDBOX_SCENARIOS
        )

        robust_success = (
            sandbox_ratio
            + FLOAT_TOLERANCE
            >= ROBUST_PASS_RATIO
        )

        feedback = (
            f"Nominal pass={nominal_success}; "
            f"sandbox pass ratio={sandbox_ratio:.3f}; "
            f"max utilization={before_max:.3f}->{after_max:.3f}"
        )

        return EvaluationResult(
            plan_generated=True,
            structurally_valid=True,
            nominal_success=nominal_success,
            sandbox_pass_ratio=sandbox_ratio,
            robust_success=robust_success,
            before_max_utilization=before_max,
            after_max_utilization=after_max,
            max_utilization_reduction=(
                before_max - after_max
            ),
            feedback=feedback,
        )



def build_verification_feedback(
    task: NetworkTask,
    plan: Optional[CandidatePlan],
    effect: EvaluationResult,
    context_message: str = "",
) -> str:
    """
    Build actionable DT feedback for Octopus.

    The feedback tells the LAM WHY the previous candidate could not be
    safely deployed, instead of only reporting a pass ratio.

    It distinguishes:
    1. no local/candidate plan;
    2. structural-constraint violation;
    3. nominal-state failure;
    4. robust-verification failure under the +5% worst-case load.
    """
    parts = []

    if context_message:
        parts.append(
            "DT analysis: " + context_message
        )

    # No plan was produced.
    if plan is None:
        parts.append(
            "No deployable local plan was generated."
        )
        parts.append(
            "The LAM should construct a new plan directly from the "
            "network state, action limit, traffic budget, and policy."
        )
        return "\n".join(parts)

    # Structural validation failed.
    if not effect.structurally_valid:
        parts.append(
            "The candidate failed structural validation: "
            + effect.feedback
        )
        parts.append(
            "Correct the violated hard constraint before optimizing "
            "the traffic amount."
        )
        return "\n".join(parts)

    # Nominal-state diagnosis.
    nominal_after = PlanEvaluator.apply_plan(
        task.loads,
        plan,
    )

    nominal_violations = []
    for link in task.loads:
        nominal_load = nominal_after[link]
        limit = (
            OVERLOAD_THRESHOLD
            * task.capacities[link]
        )

        if (
            nominal_load
            > limit + FLOAT_TOLERANCE
        ):
            nominal_violations.append(
                (
                    link,
                    nominal_load,
                    nominal_load - limit,
                )
            )

    if nominal_violations:
        parts.append(
            "The candidate does not solve the nominal network state."
        )

        for (
            link,
            load_after,
            excess,
        ) in nominal_violations:
            parts.append(
                f"- {link}: nominal post-action load "
                f"{load_after:.2f} Mbps exceeds the "
                f"{OVERLOAD_THRESHOLD * 100:.0f}% limit by "
                f"{excess:.2f} Mbps."
            )

    # Worst-case +5% diagnosis.
    worst_case_before = {
        link: (
            task.loads[link]
            * (1.0 + LOAD_PERTURBATION)
        )
        for link in task.loads
    }

    worst_case_after = (
        PlanEvaluator.apply_plan(
            worst_case_before,
            plan,
        )
    )

    robust_violations = []

    for link in task.loads:
        capacity = task.capacities[link]

        # Mandatory safety-margin boundary used by the LAM prompt.
        safe_limit = (
            OVERLOAD_THRESHOLD
            * capacity
            - SAFETY_MARGIN_MBPS
        )

        load_after = worst_case_after[link]

        if (
            load_after
            > safe_limit + FLOAT_TOLERANCE
        ):
            robust_violations.append(
                (
                    link,
                    load_after,
                    safe_limit,
                    load_after - safe_limit,
                )
            )

    parts.append(
        f"DT sandbox pass ratio: "
        f"{100.0 * effect.sandbox_pass_ratio:.1f}% "
        f"(deployment requires at least "
        f"{100.0 * ROBUST_PASS_RATIO:.1f}%)."
    )

    if robust_violations:
        parts.append(
            "Under the deterministic +5% worst-case load check, "
            "the following links still violate the safe boundary:"
        )

        for (
            link,
            load_after,
            safe_limit,
            excess,
        ) in robust_violations:
            parts.append(
                f"- {link}: worst-case post-action load "
                f"{load_after:.2f} Mbps > safe limit "
                f"{safe_limit:.2f} Mbps; reduce its net load "
                f"by at least {excess:.2f} Mbps."
            )

        parts.append(
            "Increase outgoing traffic from the violating source "
            "link(s), reduce incoming traffic to the violating target "
            "link(s), or choose different target links, while keeping "
            "the total moved traffic within the original budget."
        )

    elif not effect.robust_success:
        # Monte-Carlo failure without a deterministic +5% violation
        # can still occur near boundaries due to the exact action coupling.
        parts.append(
            "The deterministic +5% boundary check is close to feasible, "
            "but the Monte-Carlo sandbox pass ratio is still below the "
            "deployment threshold. Add more safety margin while respecting "
            "the traffic budget."
        )

    else:
        parts.append(
            "The candidate satisfies the DT deployment requirement."
        )

    return "\n".join(parts)



# ============================================================
# 8. LAM prompt and parsing
# ============================================================

def build_lam_prompt(
    task: NetworkTask,
    dt_feedback: str = "",
    previous_prompt: str = "",
    previous_output: str = "",
) -> str:
    """
    task_type is deliberately excluded.
    """
    network_state = {
        "capacities_mbps": task.capacities,
        "reported_loads_mbps": task.loads,
        "total_traffic_budget_mbps": (
            task.traffic_budget_mbps
        ),
        "maximum_number_of_actions": (
            task.max_actions
        ),
        "policy": task.policy_text,
        "maximum_allowed_link_utilization": (
            OVERLOAD_THRESHOLD
        ),
        "load_uncertainty_fraction": (
            LOAD_PERTURBATION
        ),
    }

    feedback_section = (
        dt_feedback
        if dt_feedback
        else "No previous DT verification feedback."
    )

    # For Octopus round 2, preserve the complete previous interaction.
    # This lets the LAM see exactly what it was asked, what it returned,
    # and why the DT rejected that result.
    if previous_prompt and previous_output:
        history_section = (
            "=== PREVIOUS LAM INTERACTION ===\n"
            "Previous request sent to the LAM:\n"
            + previous_prompt
            + "\n\nPrevious LAM raw output:\n"
            + previous_output
            + "\n=== END PREVIOUS LAM INTERACTION ==="
        )
    else:
        history_section = (
            "No previous LAM interaction is available. "
            "This is the first LAM request for this task."
        )

    return """
You are a network-management model.

Inspect the network state and generate a robust traffic-rerouting plan.
Infer the required number of actions from the state and constraints.

Network state:
{network_state}

Previous DT verification feedback:
{feedback_section}

Previous LAM request and result:
{history_section}

Network fluctuation model:
1. The reported loads are only instantaneous telemetry snapshots.
2. Before the generated plan is actually executed, the load of EVERY link may
   independently fluctuate within +/-5% of its reported value.
3. Therefore, do NOT design the plan only for the reported nominal loads.
4. When deciding the moved traffic, explicitly consider the worst-case +5%
   load fluctuation for both source links and target links.
5. A plan that works only for the nominal snapshot but becomes overloaded
   after a possible +5% fluctuation is NOT considered reliable.
6. Use the following worst-case constraint for EVERY link l after all actions:

   1.05 * reported_load_l
   - total_outgoing_traffic_l
   + total_incoming_traffic_l
   <= 0.90 * capacity_l - 0.50

   The 0.50 Mbps safety margin is mandatory whenever a feasible plan exists
   within the given traffic budget.

Requirements:
1. Use only the provided links.
2. Every action must use different source and target links.
3. Identify every overloaded link and relieve it.
4. If multiple links are overloaded, coordinate multiple actions when allowed.
5. Follow the natural-language policy strictly.
6. The number of actions must not exceed maximum_number_of_actions.
7. Total moved traffic over all actions must satisfy:
   0 < total moved traffic <= total_traffic_budget_mbps.
8. Under the possible +5% load fluctuation, every source link after moving
   traffic should remain at or below 90% utilization.
9. Under the possible +5% load fluctuation, every target link after receiving
   traffic should remain at or below 90% utilization.
10. Do not create a new overloaded target link.
11. First guarantee robust feasibility under the stated fluctuation; only then
    minimize the total moved traffic.
12. Before returning the answer, internally re-check the final load of every
    link under the +5% worst-case condition.
13. If DT feedback is provided, correct the previous plan accordingly.
14. Return JSON only. No Markdown, calculations, or explanations.

Required JSON:
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
        ),
        feedback_section=feedback_section,
        history_section=history_section,
    )


def extract_json(
    text: str,
) -> Dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "No JSON object found."
        )

    return json.loads(
        match.group(0)
    )


def parse_plan(
    text: str,
) -> CandidatePlan:
    data = extract_json(text)

    if "actions" in data:
        raw_actions = data["actions"]

    elif {
        "action",
        "from_link",
        "to_link",
        "traffic",
    }.issubset(set(data.keys())):
        raw_actions = [data]

    else:
        raise ValueError(
            "Output must contain an actions list."
        )

    if (
        not isinstance(raw_actions, list)
        or len(raw_actions) == 0
    ):
        raise ValueError(
            "actions must be a non-empty list."
        )

    actions = []

    for raw in raw_actions:
        required = {
            "action",
            "from_link",
            "to_link",
            "traffic",
        }

        if not required.issubset(
            set(raw.keys())
        ):
            raise ValueError(
                "An action is missing required fields."
            )

        actions.append(
            CandidateAction(
                action=str(
                    raw["action"]
                ).strip(),
                from_link=str(
                    raw["from_link"]
                ).strip(),
                to_link=str(
                    raw["to_link"]
                ).strip(),
                traffic=float(
                    raw["traffic"]
                ),
            )
        )

    return CandidatePlan(
        actions=actions
    )


# ============================================================
# 9. Real LAM API call
# ============================================================

def call_lam_once(
    task: NetworkTask,
    dt_feedback: str = "",
    previous_prompt: str = "",
    previous_output: str = "",
) -> LAMMeasurement:
    prompt = build_lam_prompt(
        task=task,
        dt_feedback=dt_feedback,
        previous_prompt=previous_prompt,
        previous_output=previous_output,
    )

    if PRINT_PROMPT:
        print(
            "\n========== LAM Prompt =========="
        )
        print(prompt)
        print(
            "================================\n"
        )

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

            # Keep consistent with previous experiments.
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = (
                chunk.choices[0].delta
            )

            content = getattr(
                delta,
                "content",
                None,
            )

            if not content:
                continue

            if first_token_time is None:
                first_token_time = (
                    time.perf_counter()
                )

            output_parts.append(
                content
            )

        end_time = time.perf_counter()

        output_text = "".join(
            output_parts
        ).strip()

        if first_token_time is None:
            ttft_ms = (
                end_time - start_time
            ) * 1000.0
        else:
            ttft_ms = (
                first_token_time
                - start_time
            ) * 1000.0

        total_latency_ms = (
            end_time - start_time
        ) * 1000.0

        try:
            plan = parse_plan(
                output_text
            )

            parse_success = True
            error_message = ""

        except Exception as parse_error:
            plan = None
            parse_success = False
            error_message = (
                f"Parse error: {parse_error}"
            )

        return LAMMeasurement(
            plan=plan,
            ttft_ms=ttft_ms,
            total_latency_ms=(
                total_latency_ms
            ),
            prompt_text=prompt,
            output_text=output_text,
            api_success=True,
            parse_success=parse_success,
            error_message=error_message,
        )

    except Exception as api_error:
        end_time = (
            time.perf_counter()
        )

        return LAMMeasurement(
            plan=None,
            ttft_ms=0.0,
            total_latency_ms=(
                end_time - start_time
            ) * 1000.0,
            prompt_text=prompt,
            output_text="",
            api_success=False,
            parse_success=False,
            error_message=(
                f"API error: {api_error}"
            ),
        )


def call_lam(
    task: NetworkTask,
    dt_feedback: str = "",
    previous_prompt: str = "",
    previous_output: str = "",
) -> LAMMeasurement:
    last = None

    for attempt in range(
        1,
        MAX_API_RETRIES + 1,
    ):
        result = call_lam_once(
            task=task,
            dt_feedback=dt_feedback,
            previous_prompt=previous_prompt,
            previous_output=previous_output,
        )

        # Every call_lam_once() executes one real
        # client.chat.completions.create(...) request.
        # Therefore, the current attempt index is the number of
        # actual LLM/API accesses consumed so far by this logical call.
        result.api_call_count = attempt
        last = result

        if result.api_success:
            return result

        print(
            f"    API attempt {attempt}/"
            f"{MAX_API_RETRIES} failed: "
            f"{result.error_message}"
        )

        if attempt < MAX_API_RETRIES:
            time.sleep(
                RETRY_INTERVAL_SECONDS
            )

    return last


# ============================================================
# 10. Architecture: DT-only
# ============================================================

def run_dt_only(
    task: NetworkTask,
    dt: DigitalTwin,
    evaluator: PlanEvaluator,
) -> ArchitectureResult:
    service_start = time.perf_counter()

    plan, decision_message = (
        dt.local_analysis(task)
    )

    # This evaluator is used to measure outcome robustness.
    # It is also counted as DT-side verification latency.
    effect = evaluator.evaluate(
        task,
        plan,
    )

    service_end = time.perf_counter()

    action_count = (
        len(plan.actions)
        if plan is not None
        else 0
    )

    total_moved = (
        sum(
            action.traffic
            for action in plan.actions
        )
        if plan is not None
        else 0.0
    )

    return ArchitectureResult(
        architecture="DT-only",
        task_id=task.task_id,
        task_type=task.task_type,
        processing_path="DT-local",

        service_latency_ms=(
            service_end - service_start
        ) * 1000.0,

        ttft_ms=0.0,

        # Unified deployable-success definition:
        # a request is successful only when the candidate is structurally
        # valid and passes the DT sandbox robustness threshold.
        task_success=int(
            effect.structurally_valid
            and effect.robust_success
        ),

        deployed=int(
            effect.structurally_valid
            and effect.robust_success
        ),

        sandbox_pass_ratio=(
            effect.sandbox_pass_ratio
        ),

        # Only an actually deployable strategy can improve the
        # physical network. Rejected candidates contribute zero.
        max_utilization_reduction_pp=(
            100.0
            * effect.max_utilization_reduction
            if (
                effect.structurally_valid
                and effect.robust_success
            )
            else 0.0
        ),

        lam_invoked=0,
        lam_calls=0,

        action_count=action_count,
        total_moved_traffic_mbps=(
            total_moved
        ),

        final_plan=plan_to_string(
            plan
        ),

        feedback=(
            decision_message
            + " | "
            + effect.feedback
        ),

        raw_lam_output="",
        error_message="",
    )


# ============================================================
# 11. Architecture: LAM-only
# ============================================================

def run_lam_only(
    task: NetworkTask,
    evaluator: PlanEvaluator,
) -> ArchitectureResult:
    service_start = time.perf_counter()

    measurement = call_lam(
        task=task,
        dt_feedback="",
    )

    # Sandbox here is used as the common offline evaluator.
    # LAM-only itself does not use the feedback for refinement.
    effect = evaluator.evaluate(
        task,
        measurement.plan,
    )

    service_end = time.perf_counter()

    plan = measurement.plan

    action_count = (
        len(plan.actions)
        if plan is not None
        else 0
    )

    total_moved = (
        sum(
            action.traffic
            for action in plan.actions
        )
        if plan is not None
        else 0.0
    )

    return ArchitectureResult(
        architecture="LAM-only",
        task_id=task.task_id,
        task_type=task.task_type,
        processing_path="LAM-direct",

        service_latency_ms=(
            service_end - service_start
        ) * 1000.0,

        ttft_ms=measurement.ttft_ms,

        # Same deployable-success criterion used by all architectures.
        # The sandbox is an offline evaluator for LAM-only; its feedback is
        # NOT returned to the LAM for refinement.
        task_success=int(
            effect.structurally_valid
            and effect.robust_success
        ),

        deployed=int(
            effect.structurally_valid
            and effect.robust_success
        ),

        sandbox_pass_ratio=(
            effect.sandbox_pass_ratio
        ),

        # Rejected LAM-only candidates are not considered deployable,
        # so they contribute zero physical-network improvement.
        max_utilization_reduction_pp=(
            100.0
            * effect.max_utilization_reduction
            if (
                effect.structurally_valid
                and effect.robust_success
            )
            else 0.0
        ),

        lam_invoked=1,

        # Count the actual number of LLM/API accesses.
        # Normally this is 1; API retries are also counted.
        lam_calls=measurement.api_call_count,

        action_count=action_count,
        total_moved_traffic_mbps=(
            total_moved
        ),

        final_plan=plan_to_string(
            plan
        ),

        feedback=effect.feedback,
        raw_lam_output=(
            measurement.output_text
        ),
        error_message=(
            measurement.error_message
        ),
    )


# ============================================================
# 12. Architecture: Octopus
# ============================================================

def run_octopus(
    task: NetworkTask,
    dt: DigitalTwin,
    evaluator: PlanEvaluator,
) -> ArchitectureResult:
    service_start = time.perf_counter()

    # --------------------------------------------------------
    # Stage 1: Local DT
    # --------------------------------------------------------
    local_plan, local_message = (
        dt.local_analysis(task)
    )

    local_effect = evaluator.evaluate(
        task,
        local_plan,
    )

    if (
        local_effect.structurally_valid
        and local_effect.robust_success
    ):
        service_end = time.perf_counter()

        return ArchitectureResult(
            architecture="Octopus",
            task_id=task.task_id,
            task_type=task.task_type,
            processing_path="DT-local",

            service_latency_ms=(
                service_end - service_start
            ) * 1000.0,

            ttft_ms=0.0,

            task_success=1,
            deployed=1,

            sandbox_pass_ratio=(
                local_effect.sandbox_pass_ratio
            ),

            max_utilization_reduction_pp=(
                100.0
                * local_effect.max_utilization_reduction
            ),

            lam_invoked=0,
            lam_calls=0,

            action_count=(
                len(local_plan.actions)
            ),

            total_moved_traffic_mbps=sum(
                action.traffic
                for action
                in local_plan.actions
            ),

            final_plan=plan_to_string(
                local_plan
            ),

            feedback=(
                local_message
                + " | "
                + local_effect.feedback
            ),

            raw_lam_output="",
            error_message="",
        )

    # --------------------------------------------------------
    # Stage 2: Escalation to LAM
    # --------------------------------------------------------
    # The local DT did not produce a safely deployable result.
    # Give the LAM an explicit diagnosis instead of only saying "failed".
    feedback = build_verification_feedback(
        task=task,
        plan=local_plan,
        effect=local_effect,
        context_message=local_message,
    )

    final_measurement = None
    final_effect = None
    final_plan = None

    actual_lam_calls = 0
    total_ttft_ms = 0.0

    # Full audit trail for all Octopus LAM rounds.
    interaction_records = []
    errors = []

    # These are empty for round 1. After round 1, they are passed
    # verbatim into round 2.
    previous_prompt = ""
    previous_output = ""

    for round_index in range(
        1,
        MAX_LAM_CALLS_PER_OCTOPUS_REQUEST + 1,
    ):
        measurement = call_lam(
            task=task,
            dt_feedback=feedback,
            previous_prompt=previous_prompt,
            previous_output=previous_output,
        )

        # Count actual LLM/API accesses, not merely the number of
        # tasks that invoked the LAM and not merely the number of rounds.
        #
        # Examples:
        #   one successful LAM request            -> +1
        #   two Octopus refinement rounds         -> +2
        #   round 1 retries once + round 2 once   -> +3
        actual_lam_calls += measurement.api_call_count

        final_measurement = measurement

        total_ttft_ms += (
            measurement.ttft_ms
        )

        if measurement.error_message:
            errors.append(
                f"Round {round_index}: "
                f"{measurement.error_message}"
            )

        effect = evaluator.evaluate(
            task,
            measurement.plan,
        )

        final_effect = effect
        final_plan = measurement.plan

        # ----------------------------------------------------
        # Print the complete Octopus LAM interaction so the
        # experiment can be audited directly from the console.
        # ----------------------------------------------------
        print(
            "\n"
            + "=" * 72
        )
        print(
            f"OCTOPUS LAM ROUND {round_index} "
            f"(actual API accesses in this round: "
            f"{measurement.api_call_count})"
        )
        print(
            "=" * 72
        )

        print(
            "\n[REQUEST / PROMPT SENT TO LAM]"
        )
        print(
            measurement.prompt_text
        )

        print(
            "\n[LAM RAW OUTPUT]"
        )
        print(
            measurement.output_text
            if measurement.output_text
            else "<empty output>"
        )

        print(
            "\n[PARSED PLAN]"
        )
        print(
            plan_to_string(
                measurement.plan
            )
        )

        print(
            "\n[DT VERIFICATION RESULT]"
        )
        print(
            effect.feedback
        )
        print(
            f"Deployable: "
            f"{bool(effect.structurally_valid and effect.robust_success)}"
        )

        # Generate the detailed DT diagnosis now, even if this is the
        # last round, so it is available in the audit record.
        round_feedback = build_verification_feedback(
            task=task,
            plan=measurement.plan,
            effect=effect,
            context_message=(
                f"LAM round {round_index} candidate "
                f"{'passed' if (effect.structurally_valid and effect.robust_success) else 'failed'} "
                f"the DT deployment check."
            ),
        )

        print(
            "\n[DT DETAILED FEEDBACK]"
        )
        print(
            round_feedback
        )
        print(
            "=" * 72
            + "\n"
        )

        # Save a complete text record to detailed CSV.
        interaction_records.append(
            (
                f"===== OCTOPUS LAM ROUND {round_index} =====\n"
                f"[PROMPT]\n{measurement.prompt_text}\n\n"
                f"[LAM RAW OUTPUT]\n{measurement.output_text}\n\n"
                f"[PARSED PLAN]\n{plan_to_string(measurement.plan)}\n\n"
                f"[DT VERIFICATION]\n{effect.feedback}\n\n"
                f"[DT DETAILED FEEDBACK]\n{round_feedback}\n"
            )
        )

        # Pass -> deploy immediately.
        if (
            effect.structurally_valid
            and effect.robust_success
        ):
            break

        # ----------------------------------------------------
        # Prepare round 2.
        #
        # IMPORTANT:
        # The next prompt contains ALL THREE:
        #   1. round-1 full request/prompt,
        #   2. round-1 raw LAM result,
        #   3. round-1 detailed DT rejection reason.
        # ----------------------------------------------------
        feedback = round_feedback
        previous_prompt = measurement.prompt_text
        previous_output = measurement.output_text

    service_end = time.perf_counter()

    if final_effect is None:
        raise RuntimeError(
            "Octopus executed no LAM round."
        )

    deployable = (
        final_effect.structurally_valid
        and final_effect.robust_success
    )

    action_count = (
        len(final_plan.actions)
        if final_plan is not None
        else 0
    )

    total_moved = (
        sum(
            action.traffic
            for action
            in final_plan.actions
        )
        if final_plan is not None
        else 0.0
    )

    return ArchitectureResult(
        architecture="Octopus",
        task_id=task.task_id,
        task_type=task.task_type,
        processing_path="LAM-escalation",

        service_latency_ms=(
            service_end - service_start
        ) * 1000.0,

        ttft_ms=total_ttft_ms,

        # Unified deployable-success definition used by all architectures.
        task_success=int(
            deployable
        ),

        deployed=int(
            deployable
        ),

        sandbox_pass_ratio=(
            final_effect.sandbox_pass_ratio
        ),

        # Count physical-network improvement only when the final plan
        # is actually deployed. A rejected candidate produces no real
        # network-state improvement.
        max_utilization_reduction_pp=(
            100.0
            * final_effect.max_utilization_reduction
            if deployable
            else 0.0
        ),

        lam_invoked=1,
        lam_calls=(
            actual_lam_calls
        ),

        action_count=action_count,
        total_moved_traffic_mbps=(
            total_moved
        ),

        final_plan=plan_to_string(
            final_plan
        ),

        feedback=(
            final_effect.feedback
        ),

        # Store the full request-output-verification history, not only
        # the raw LAM outputs. This makes the second-round refinement
        # completely observable from the detailed CSV.
        raw_lam_output="\n\n".join(
            interaction_records
        ),

        error_message="\n".join(
            errors
        ),
    )


# ============================================================
# 13. Five main comparison metrics
#
# Unified definitions:
# - Deployable Success Rate:
#   structurally valid + sandbox pass ratio >= ROBUST_PASS_RATIO
#   for ALL three architectures.
# - Successful-Strategy Robustness:
#   average sandbox pass ratio ONLY among safely deployable strategies.
# - Avg. Max-Util. Reduction:
#   physical-network improvement ONLY from deployable strategies;
#   rejected candidates contribute 0 for all architectures.
# - Total LAM Calls:
#   counts EVERY actual client.chat.completions.create(...) access.
#   It is NOT a binary "whether the task invoked LAM" metric.
#   For example, an Octopus task using two LAM rounds contributes 2 calls.
#   If an API request is retried, the retry is also counted as another call.

# ============================================================

def summarize_architecture(
    architecture: str,
    rows: List[ArchitectureResult],
) -> Dict[str, Any]:
    successful_rows = [
        row
        for row in rows
        if row.task_success == 1
    ]

    successful_strategy_robustness = (
        100.0 * mean(
            row.sandbox_pass_ratio
            for row in successful_rows
        )
        if successful_rows
        else 0.0
    )

    return {
        "Architecture": architecture,

        "Average Latency (ms)": round(
            mean(
                row.service_latency_ms
                for row in rows
            ),
            2,
        ),

        "Deployable Success Rate (%)": round(
            100.0 * mean(
                row.task_success
                for row in rows
            ),
            2,
        ),

        "Successful-Strategy Robustness (%)": round(
            successful_strategy_robustness,
            2,
        ),

        "Avg. Max-Util. Reduction (pp)": round(
            mean(
                row.max_utilization_reduction_pp
                for row in rows
            ),
            2,
        ),

        "Total LAM Calls": sum(
            row.lam_calls
            for row in rows
        ),
    }


# ============================================================
# 14. Save helpers
# ============================================================

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
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 15. Main experiment
# ============================================================

def main():
    tasks = generate_common_tasks(
        num_tasks=NUM_TASKS,
        seed=RANDOM_SEED,
    )

    routine_count = sum(
        task.task_type == "Routine"
        for task in tasks
    )

    multi_count = sum(
        task.task_type == "Multi-link"
        for task in tasks
    )

    policy_count = sum(
        task.task_type == "Policy"
        for task in tasks
    )

    print(
        "Common workload:"
        f" {routine_count} Routine,"
        f" {multi_count} Multi-link,"
        f" {policy_count} Policy"
    )

    dt = DigitalTwin()
    evaluator = PlanEvaluator()

    all_results = []

    # ========================================================
    # A. DT-only
    # ========================================================

    print(
        "\n"
        "========================================"
    )
    print("Running DT-only")
    print(
        "========================================"
    )

    for index, task in enumerate(
        tasks,
        start=1,
    ):
        print(
            f"\n[DT-only] "
            f"Task {index}/{len(tasks)}"
            f" | Label={task.task_type}"
        )

        print(
            f"  Loads: {task.loads}"
        )

        result = run_dt_only(
            task,
            dt,
            evaluator,
        )

        all_results.append(
            result
        )

        print(
            f"  Plan: {result.final_plan}"
        )
        print(
            f"  Deployable success: {result.task_success}"
        )
        print(
            f"  Robustness: "
            f"{100.0 * result.sandbox_pass_ratio:.2f}%"
        )

    # ========================================================
    # B. LAM-only
    # ========================================================

    print(
        "\n"
        "========================================"
    )
    print("Running LAM-only")
    print(
        "========================================"
    )

    for index, task in enumerate(
        tasks,
        start=1,
    ):
        print(
            f"\n[LAM-only] "
            f"Task {index}/{len(tasks)}"
            f" | Label={task.task_type}"
        )

        print(
            f"  Loads: {task.loads}"
        )

        result = run_lam_only(
            task,
            evaluator,
        )

        all_results.append(
            result
        )

        print(
            f"  Plan: {result.final_plan}"
        )
        print(
            f"  Deployable success: {result.task_success}"
        )
        print(
            f"  Robustness: "
            f"{100.0 * result.sandbox_pass_ratio:.2f}%"
        )
        print(
            f"  Latency: "
            f"{result.service_latency_ms:.2f} ms"
        )

    # ========================================================
    # C. Octopus
    # ========================================================

    print(
        "\n"
        "========================================"
    )
    print("Running Octopus")
    print(
        "========================================"
    )

    for index, task in enumerate(
        tasks,
        start=1,
    ):
        print(
            f"\n[Octopus] "
            f"Task {index}/{len(tasks)}"
            f" | Label={task.task_type}"
        )

        print(
            f"  Loads: {task.loads}"
        )

        result = run_octopus(
            task,
            dt,
            evaluator,
        )

        all_results.append(
            result
        )

        print(
            f"  Path: {result.processing_path}"
        )
        print(
            f"  LAM calls: {result.lam_calls}"
        )
        print(
            f"  Plan: {result.final_plan}"
        )
        print(
            f"  Deployable success: {result.task_success}"
        )
        print(
            f"  Deployed after DT verification: {result.deployed}"
        )
        print(
            f"  Robustness: "
            f"{100.0 * result.sandbox_pass_ratio:.2f}%"
        )
        print(
            f"  Latency: "
            f"{result.service_latency_ms:.2f} ms"
        )

    # ========================================================
    # Summary
    # ========================================================

    summaries = []

    for architecture in [
        "DT-only",
        "LAM-only",
        "Octopus",
    ]:
        rows = [
            row
            for row in all_results
            if row.architecture
            == architecture
        ]

        summaries.append(
            summarize_architecture(
                architecture,
                rows,
            )
        )

    print(
        "\n"
        "========================================"
    )
    print("FINAL COMPARISON")
    print(
        "========================================"
    )

    for summary in summaries:
        print(
            f"\nArchitecture: "
            f"{summary['Architecture']}"
        )

        for key, value in summary.items():
            if key == "Architecture":
                continue

            print(
                f"{key}: {value}"
            )

    # Detailed CSV
    save_csv(
        "three_architecture_detailed_results.csv",
        [
            asdict(result)
            for result in all_results
        ],
    )

    # Summary CSV
    save_csv(
        "three_architecture_summary.csv",
        summaries,
    )

    print("\nSaved:")
    print(
        "  three_architecture_detailed_results.csv"
    )
    print(
        "  three_architecture_summary.csv"
    )


if __name__ == "__main__":
    main()






