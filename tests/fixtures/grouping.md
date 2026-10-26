# Grouping rows

## GROUP BY

The GROUP BY clause combines rows that have the same values in the grouping columns into one result row per group. Aggregate functions such as count(*), avg(X) and sum(X) are then computed once for every group.

```sql
SELECT city, count(*)
FROM customers
GROUP BY city;
```

## HAVING

The HAVING clause filters groups after the aggregates are computed. Groups for which the HAVING expression is false are removed from the result. This is different from WHERE, which filters individual rows before grouping.

- count(*) counts all rows in the group.
- count(X) counts the rows where X is not NULL.
