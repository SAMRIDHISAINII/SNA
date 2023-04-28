LOAD CSV FROM "file:///terrorist_data_all.csv" AS row return row
MATCH(N) RETURN COUNT(N)
MATCH (n)-->() 
RETURN id(n),count(*) as degree 

ORDER BY degree DESC LIMIT 10
MATCH (n)

WHERE rand() <= 0.5
WITH n LIMIT 1000
MATCH (n)-[r]-(m)
RETURN n, r, m
MATCH (n)
WITH n LIMIT 1000
MATCH (n)-[r]-(m)
RETURN n, r, m
