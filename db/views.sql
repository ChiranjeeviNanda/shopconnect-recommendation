-- Drop views in reverse dependency order for clean schema resets
DROP VIEW IF EXISTS v_household_repeat_intervals CASCADE;

DROP VIEW IF EXISTS v_template_goal_conversion CASCADE;

DROP VIEW IF EXISTS v_common_swap_pairs CASCADE;

DROP VIEW IF EXISTS v_recipe_performance CASCADE;

-- 1. Recipe & Meal Role Performance Metrics
CREATE VIEW v_recipe_performance AS
SELECT
    r.recipe_code,
    r.name AS recipe_name,
    r.meal_role,
    r.diet,
    COUNT(i.interaction_id) AS total_interactions,
    ROUND(
        SUM(
            CASE
                WHEN i.action = 'accepted' THEN 1
                ELSE 0
            END
        ) * 100.0 / NULLIF(COUNT(i.interaction_id), 0),
        2
    ) AS acceptance_rate_pct,
    ROUND(
        SUM(
            CASE
                WHEN i.action = 'swapped' THEN 1
                ELSE 0
            END
        ) * 100.0 / NULLIF(COUNT(i.interaction_id), 0),
        2
    ) AS swap_rate_pct,
    ROUND(
        SUM(
            CASE
                WHEN i.action = 'skipped' THEN 1
                ELSE 0
            END
        ) * 100.0 / NULLIF(COUNT(i.interaction_id), 0),
        2
    ) AS skip_rate_pct,
    ROUND(
        SUM(
            CASE
                WHEN i.action = 'cooked' THEN 1
                ELSE 0
            END
        ) * 100.0 / NULLIF(COUNT(i.interaction_id), 0),
        2
    ) AS cook_rate_pct
FROM recipes r
    LEFT JOIN interactions i ON r.recipe_code = i.recipe_code
GROUP BY
    r.recipe_code,
    r.name,
    r.meal_role,
    r.diet;

-- 2. Most Common Swap Pairs (A -> B Transitions)
CREATE VIEW v_common_swap_pairs AS
SELECT
    i.recipe_code AS original_recipe_code,
    r1.name AS original_recipe_name,
    r1.meal_role AS original_meal_role,
    i.swapped_to_recipe_code,
    r2.name AS swapped_to_recipe_name,
    r2.meal_role AS swapped_to_meal_role,
    COUNT(*) AS transition_count
FROM
    interactions i
    JOIN recipes r1 ON i.recipe_code = r1.recipe_code
    JOIN recipes r2 ON i.swapped_to_recipe_code = r2.recipe_code
WHERE
    i.action = 'swapped'
    AND i.swapped_to_recipe_code IS NOT NULL
GROUP BY
    i.recipe_code,
    r1.name,
    r1.meal_role,
    i.swapped_to_recipe_code,
    r2.name,
    r2.meal_role
ORDER BY transition_count DESC;

-- 3. Plan-to-Order Conversion Rate Grouped by Template Goal
CREATE VIEW v_template_goal_conversion AS
SELECT
    mp.template_goal,
    COUNT(DISTINCT mp.plan_id) AS total_plans_created,
    COUNT(DISTINCT o.order_id) AS total_orders_placed,
    ROUND(
        COUNT(DISTINCT o.order_id) * 100.0 / NULLIF(COUNT(DISTINCT mp.plan_id), 0),
        2
    ) AS conversion_rate_pct
FROM meal_plans mp
    LEFT JOIN orders o ON mp.plan_id = o.plan_id
GROUP BY
    mp.template_goal
ORDER BY conversion_rate_pct DESC;

-- 4. Repeat-Dish Intervals per Household
CREATE VIEW v_household_repeat_intervals AS
WITH ordered_plan_items AS (
    SELECT 
        mp.household_id,
        mpi.recipe_code,
        mpi.date,
        LAG(mpi.date) OVER (PARTITION BY mp.household_id, mpi.recipe_code ORDER BY mpi.date) AS prev_date
    FROM meal_plan_items mpi
    JOIN meal_plans mp ON mpi.plan_id = mp.plan_id
),
interval_calc AS (
    SELECT 
        household_id,
        recipe_code,
        date - prev_date AS days_between_repeats
    FROM ordered_plan_items
    WHERE prev_date IS NOT NULL
)
SELECT 
    household_id,
    ROUND(AVG(days_between_repeats)::numeric, 1) AS avg_repeat_interval_days,
    MIN(days_between_repeats) AS min_repeat_interval_days,
    COUNT(*) AS total_repeats_tracked
FROM interval_calc
GROUP BY household_id;