SELECT r.return_id,
       r.order_id,
       r.product_id,
       r.reason,
       r.created_at
FROM returns r
ORDER BY r.created_at DESC
LIMIT 100;
