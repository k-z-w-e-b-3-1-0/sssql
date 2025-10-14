SELECT DATE_TRUNC('month', order_date) AS month,
       COUNT(*) AS order_count,
       SUM(total_amount) AS total_sales
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
