SELECT cancellation_reason,
       COUNT(*) AS cancellation_count
FROM order_cancellations
GROUP BY cancellation_reason
ORDER BY cancellation_count DESC;
