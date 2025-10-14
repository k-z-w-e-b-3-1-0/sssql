SELECT user_id,
       COUNT(*) AS login_count,
       MAX(logged_in_at) AS last_login
FROM user_sessions
GROUP BY user_id
ORDER BY login_count DESC;
