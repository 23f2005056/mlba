import duckdb
con = duckdb.connect()
rows = con.execute("""
  SELECT specialty, MIN(cms_year), MAX(cms_year), COUNT(DISTINCT cms_year), COUNT(*)
  FROM read_parquet('data/processed/provider_year.parquet')
  GROUP BY 1 HAVING COUNT(DISTINCT cms_year) < 7 ORDER BY 5 DESC LIMIT 25""").fetchall()
for r in rows: print(r)