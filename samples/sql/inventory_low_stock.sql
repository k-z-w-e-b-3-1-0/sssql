SELECT product_id,
       product_name,
       quantity_on_hand
FROM inventory
WHERE quantity_on_hand < reorder_threshold
ORDER BY quantity_on_hand ASC;
