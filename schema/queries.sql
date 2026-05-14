-- Sample analytical queries for academic reporting

-- 1) City-wise average AQI
SELECT city, AVG(aqi) AS avg_aqi
FROM processed_air_quality
GROUP BY city
ORDER BY avg_aqi DESC;

-- 2) AQI category distribution
SELECT aqi_category, COUNT(*) AS records
FROM processed_air_quality
GROUP BY aqi_category
ORDER BY records DESC;

-- 3) Weekly pattern: average AQI by day_of_week
SELECT day_of_week, AVG(aqi) AS avg_aqi
FROM processed_air_quality
GROUP BY day_of_week
ORDER BY day_of_week;

-- 4) Trend: daily average PM2.5 by city
SELECT city, DATE(date) AS day, AVG(pm2_5) AS avg_pm25
FROM processed_air_quality
GROUP BY city, DATE(date)
ORDER BY city, day;
