// Run this before any data ingestion to prevent duplicate nodes.
// MERGE without constraints silently creates duplicates when the same
// instrument appears in multiple rows.

CREATE CONSTRAINT IF NOT EXISTS FOR (i:Instrument) REQUIRE i.ticker IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (f:Fund) REQUIRE f.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Trade) REQUIRE t.trade_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Analyst) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE;
