// Define the OLAP query
WITH ['CategoryA', 'CategoryB', 'CategoryC'] AS categories
MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
WHERE c.name IN categories
RETURN c.name, COUNT(p) AS count

// Run the OLAP query on the cluster
CALL ga.cluster.olap.run({
  cypher: '
    WITH $categories AS categories
    MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
    WHERE c.name IN categories
    RETURN c.name, COUNT(p) AS count
  ',
  params: {
    categories: ['CategoryA', 'CategoryB', 'CategoryC']
  }
}) YIELD result
RETURN result
