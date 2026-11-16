import random

GOAL_CONVERSION_PROBS = {
    'weight-loss': 0.92,
    'high-protein': 0.85,
    'everyday': 0.72
}


def should_generate_order(h_goal, plan_interactions, plan_has_interaction, is_cold_start):
    """Evaluates whether a household converts a weekly meal plan into a grocery order."""
    if not plan_has_interaction or is_cold_start:
        return False

    base_conversion_prob = GOAL_CONVERSION_PROBS.get(h_goal, 0.80)

    # Adjust conversion probability downward if the household skipped many meals
    if plan_interactions:
        skip_ratio = sum(1 for i in plan_interactions if i['action'] == 'skipped') / len(plan_interactions)
        base_conversion_prob *= (1.0 - (skip_ratio * 0.5))

    return random.random() < base_conversion_prob