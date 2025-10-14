SELECT customer_id,
       first_name,
       last_name,
       status
FROM customers
WHERE status = 'active'
ORDER BY last_name, first_name;
