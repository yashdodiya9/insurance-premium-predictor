-- Analytics over the prediction_logs table (PostgreSQL).
-- Run with:  psql "$DATABASE_URL" -f sql/analytics.sql

-- 1. How many predictions per category, and how confident is the model on each?
SELECT predicted_category,
       COUNT(*)                          AS predictions,
       ROUND(AVG(confidence)::numeric, 3) AS avg_confidence,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
FROM prediction_logs
GROUP BY predicted_category
ORDER BY predictions DESC;

-- 2. Category mix by city tier (which kinds of cities get which premium class).
SELECT city_tier,
       COUNT(*) FILTER (WHERE predicted_category = 'Low')    AS low,
       COUNT(*) FILTER (WHERE predicted_category = 'Medium') AS medium,
       COUNT(*) FILTER (WHERE predicted_category = 'High')   AS high,
       COUNT(*)                                              AS total
FROM prediction_logs
GROUP BY city_tier
ORDER BY city_tier;

-- 3. Average BMI and income by lifestyle risk, only groups with at least 5 requests.
SELECT lifestyle_risk,
       COUNT(*)                         AS n,
       ROUND(AVG(bmi)::numeric, 1)       AS avg_bmi,
       ROUND(AVG(income_lpa)::numeric, 1) AS avg_income_lpa
FROM prediction_logs
GROUP BY lifestyle_risk
HAVING COUNT(*) >= 5
ORDER BY avg_bmi DESC;

-- 4. Daily prediction volume with a running total (date_trunc + window function).
SELECT day,
       requests,
       SUM(requests) OVER (ORDER BY day) AS cumulative_requests
FROM (
    SELECT date_trunc('day', created_at)::date AS day,
           COUNT(*)                            AS requests
    FROM prediction_logs
    GROUP BY 1
) d
ORDER BY day;

-- 5. Low-confidence predictions (below 0.6) ranked within each category.
SELECT predicted_category, confidence, age_grp, lifestyle_risk, occupation,
       RANK() OVER (PARTITION BY predicted_category ORDER BY confidence) AS rank_in_category
FROM prediction_logs
WHERE confidence < 0.6
ORDER BY predicted_category, rank_in_category
LIMIT 20;
