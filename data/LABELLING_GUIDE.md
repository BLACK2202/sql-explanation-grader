# Labelling guide

Label each explanation good or bad.
An explanation is good only if ALL are true:

1. Correct: it says what the query really returns, with no wrong claims.
2. Complete: it covers every join, filter, grouping and ordering.
3. Clear: a non-technical reader can follow it, and SQL terms are explained.
   If any criterion fails, the label is bad.

## Examples

Good: `SELECT COUNT(*) FROM users WHERE age > 30;` means the query counts users whose age is greater than 30.
Reason: It identifies the counted rows and explains the `WHERE` condition accurately.

Good: `SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY COUNT(*) DESC;` lists each city and its customer count, from the most customers to the fewest.
Reason: It covers the selected columns, grouping, aggregate, and ordering.

Good: `SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL;` lists customers who have no matching orders.
Reason: It explains the left join and the null filter, including the resulting population.

Bad: `SELECT COUNT(*) FROM users WHERE age > 30;` lists every user older than 30.
Reason: The query returns one count, not the individual users.

Bad: `SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY COUNT(*) DESC;` lists cities alphabetically.
Reason: The query orders by the count in descending order, not by city name.

Bad: `SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL;` lists customers and all their orders.
Reason: The `IS NULL` filter keeps only customers without a matching order.
