-- Drop tables in reverse dependency order for clean schema resets
DROP TABLE IF EXISTS orders CASCADE;

DROP TABLE IF EXISTS interactions CASCADE;

DROP TABLE IF EXISTS meal_plan_items CASCADE;

DROP TABLE IF EXISTS meal_plans CASCADE;

DROP TABLE IF EXISTS food_dislikes CASCADE;

DROP TABLE IF EXISTS households CASCADE;

DROP TABLE IF EXISTS recipes CASCADE;

-- 1. Recipes Table (Fully mapped to recipe_list.csv analysis)
CREATE TABLE recipes (
    id INT, -- Original sequential numeric ID
    recipe_code VARCHAR(64) PRIMARY KEY, -- Unique code (max length ~56)
    name VARCHAR(64) NOT NULL, -- Recipe name (max length ~46)
    description TEXT, -- Detailed description (~782 chars)
    status VARCHAR(16) NOT NULL, -- e.g., published/active status
    is_public BOOLEAN, -- 't'/'f' boolean indicator
    is_active BOOLEAN, -- 't'/'f' boolean indicator
    course VARCHAR(32), -- e.g., {side}, {main}
    meal_role VARCHAR(32), -- anchor, main, lentil, condiment, side, etc.
    dish_family VARCHAR(32), -- curry, dal, sabzi, dosa, etc.
    meal_types VARCHAR(64), -- Breakfast, lunch, dinner applicability
    cuisines TEXT, -- Formatted set string e.g., '{KARNATAKA,TAMIL_NADU}'
    region VARCHAR(32), -- south_india, north_india, west_india, east_india
    diet VARCHAR(32), -- vegetarian, non-vegetarian, etc.
    spice_level VARCHAR(16), -- mild, medium, spicy
    allergen_tags VARCHAR(64), -- Allergen flags
    allergens_confirmed CHAR(1),
    texture VARCHAR(64),
    health_tags VARCHAR(64), -- e.g., '{gluten_free}'
    efficiency_tags VARCHAR(64),
    experience_tags VARCHAR(128),
    cost_tier VARCHAR(32), -- budget_friendly, mid_range, premium
    difficulty_level VARCHAR(16), -- easy, medium, hard
    servings INT DEFAULT 1,
    prep_time_min INT DEFAULT 0,
    cook_time_min INT DEFAULT 0,
    total_time_min FLOAT DEFAULT 0.0,
    passive_time_min FLOAT DEFAULT 0.0,
    ingredients_json TEXT NOT NULL, -- JSON payload of ingredients (~3907 chars max)
    prep_steps TEXT, -- Step-by-step prep instructions
    cook_steps TEXT, -- Cooking instructions (~1924 chars max)
    nutrition_info VARCHAR(128),
    tags TEXT, -- General tag bag
    metadata TEXT, -- Additional metadata blob
    image_url VARCHAR(255),
    youtube_url VARCHAR(128),
    source VARCHAR(64),
    language VARCHAR(8),
    created_by VARCHAR(32),
    approved_by VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP,
    popularity_score FLOAT DEFAULT 0.0 -- Derived via synthetic interactions
);

-- 2. Households Table (Synthetic)
CREATE TABLE households (
    household_id VARCHAR(50) PRIMARY KEY,
    household_size INT NOT NULL,
    home_region VARCHAR(64) NOT NULL, -- Matches recipe regions (south_india, north_india, etc.)
    diet VARCHAR(32) NOT NULL, -- vegetarian, non-vegetarian, vegan
    spice_level VARCHAR(16) NOT NULL, -- mild, medium, spicy
    health_goal VARCHAR(32) NOT NULL, -- everyday, high-protein, weight-loss
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Food Dislikes Table
CREATE TABLE food_dislikes (
    dislike_id SERIAL PRIMARY KEY,
    household_id VARCHAR(50) REFERENCES households (household_id) ON DELETE CASCADE,
    ingredient_or_dish VARCHAR(255) NOT NULL
);

-- 4. Meal Plans Table
CREATE TABLE meal_plans (
    plan_id VARCHAR(50) PRIMARY KEY,
    household_id VARCHAR(50) REFERENCES households (household_id) ON DELETE CASCADE,
    template_goal VARCHAR(32) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    version INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Meal Plan Items Table
CREATE TABLE meal_plan_items (
    item_id SERIAL PRIMARY KEY,
    plan_id VARCHAR(50) REFERENCES meal_plans (plan_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    slot VARCHAR(20) NOT NULL, -- breakfast, lunch, dinner
    recipe_code VARCHAR(64) REFERENCES recipes (recipe_code),
    meal_role VARCHAR(32) NOT NULL,
    UNIQUE (plan_id, date, slot) -- Prevents duplicate recipe slots on the same day
);

-- 6. Interactions Table
CREATE TABLE interactions (
    interaction_id SERIAL PRIMARY KEY,
    household_id VARCHAR(50) REFERENCES households (household_id) ON DELETE CASCADE,
    plan_id VARCHAR(50) REFERENCES meal_plans (plan_id) ON DELETE SET NULL,
    recipe_code VARCHAR(64) REFERENCES recipes (recipe_code),
    date DATE NOT NULL,
    action VARCHAR(20) NOT NULL, -- accepted, swapped, skipped, cooked
    swapped_to_recipe_code VARCHAR(64) REFERENCES recipes (recipe_code),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Orders Table
CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    household_id VARCHAR(50) REFERENCES households (household_id) ON DELETE CASCADE,
    plan_id VARCHAR(50) REFERENCES meal_plans (plan_id) ON DELETE SET NULL,
    source VARCHAR(32) NOT NULL, -- meal_plan, manual
    order_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance Indexes
CREATE INDEX idx_recipes_region ON recipes (region);

CREATE INDEX idx_recipes_diet ON recipes (diet);

CREATE INDEX idx_recipes_meal_role ON recipes (meal_role);

CREATE INDEX idx_interactions_household_date ON interactions (household_id, date);

CREATE INDEX idx_meal_plan_items_plan ON meal_plan_items (plan_id, date);