SELECT e.employee_id,
       e.full_name,
       AVG(r.rating) AS average_rating
FROM employees e
JOIN performance_reviews r ON r.employee_id = e.employee_id
GROUP BY e.employee_id, e.full_name
HAVING COUNT(r.review_id) >= 3
ORDER BY average_rating DESC;
