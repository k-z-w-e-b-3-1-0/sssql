SELECT region,
       SUM(total_amount) AS total_revenue
FROM orders
GROUP BY region
ORDER BY total_revenue DESC;
