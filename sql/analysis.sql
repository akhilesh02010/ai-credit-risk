-- Each query is intentionally aggregate-only. The public source contains no names or contacts.

-- name: portfolio_summary
SELECT
    COUNT(*) AS customers,
    SUM("DEFAULT") AS defaults,
    ROUND(100.0 * AVG("DEFAULT"), 2) AS default_rate_pct,
    ROUND(AVG(LIMIT_BAL), 2) AS avg_credit_limit,
    ROUND(AVG(AGE), 1) AS avg_age
FROM customer_credit;

-- name: september_payment_status_risk
SELECT
    PAY_SEPT AS payment_status,
    COUNT(*) AS customers,
    SUM("DEFAULT") AS defaults,
    ROUND(100.0 * AVG("DEFAULT"), 2) AS default_rate_pct
FROM customer_credit
GROUP BY PAY_SEPT
ORDER BY PAY_SEPT;

-- name: credit_limit_band_risk
SELECT
    CASE
        WHEN LIMIT_BAL < 100000 THEN '<100k'
        WHEN LIMIT_BAL < 200000 THEN '100k-199k'
        WHEN LIMIT_BAL < 300000 THEN '200k-299k'
        WHEN LIMIT_BAL < 500000 THEN '300k-499k'
        ELSE '500k+'
    END AS limit_band,
    COUNT(*) AS customers,
    ROUND(AVG(LIMIT_BAL), 0) AS avg_limit,
    ROUND(100.0 * AVG("DEFAULT"), 2) AS default_rate_pct
FROM customer_credit
GROUP BY limit_band
ORDER BY MIN(LIMIT_BAL);

-- name: age_band_risk
SELECT
    CASE
        WHEN AGE < 30 THEN '<30'
        WHEN AGE < 40 THEN '30-39'
        WHEN AGE < 50 THEN '40-49'
        WHEN AGE < 60 THEN '50-59'
        ELSE '60+'
    END AS age_band,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG("DEFAULT"), 2) AS default_rate_pct,
    ROUND(AVG(LIMIT_BAL), 0) AS avg_credit_limit
FROM customer_credit
GROUP BY age_band
ORDER BY MIN(AGE);

-- name: delinquency_breadth_risk
WITH delinquency AS (
    SELECT *,
        (PAY_SEPT > 0) + (PAY_AUG > 0) + (PAY_JUL > 0) +
        (PAY_JUN > 0) + (PAY_MAY > 0) + (PAY_APR > 0) AS delinquent_months
    FROM customer_credit
)
SELECT
    delinquent_months,
    COUNT(*) AS customers,
    SUM("DEFAULT") AS defaults,
    ROUND(100.0 * AVG("DEFAULT"), 2) AS default_rate_pct
FROM delinquency
GROUP BY delinquent_months
ORDER BY delinquent_months;
