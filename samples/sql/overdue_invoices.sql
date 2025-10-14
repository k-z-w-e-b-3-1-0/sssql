SELECT invoice_id,
       customer_id,
       due_date,
       total_due
FROM invoices
WHERE paid_at IS NULL
  AND due_date < CURRENT_DATE
ORDER BY due_date;
