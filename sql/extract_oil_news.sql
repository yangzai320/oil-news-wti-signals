-- Extract oil-relevant headlines from the GDELT GKG 2.1 table (2020-2025) and
-- export them to Cloud Storage as sharded CSVs.
--
-- Pipeline: parse timestamp -> pull <PAGE_TITLE> from Extras -> require an oil
-- anchor term AND at least one market-driver group -> drop exact duplicates.
-- Replace the bucket URI with your own before running.

EXPORT DATA OPTIONS(
  uri='gs://YOUR_BUCKET/oil_news_*.csv',
  format='CSV',
  overwrite=true,
  header=true
)
AS
WITH base AS (
  SELECT
    TIMESTAMP_TRUNC(
      TIMESTAMP(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING))),
      MINUTE
    ) AS news_time,

    REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS raw_title,

    COALESCE(V2Themes, Themes) AS theme_keywords,
    IFNULL(V2Themes, '') AS v2themes,
    IFNULL(Themes, '') AS themes,
    IFNULL(V2Locations, '') AS v2locations,
    IFNULL(Locations, '') AS locations,
    IFNULL(V2Organizations, '') AS v2organizations,
    IFNULL(Organizations, '') AS organizations,
    IFNULL(V2Persons, '') AS v2persons,
    IFNULL(Persons, '') AS persons,
    IFNULL(DocumentIdentifier, '') AS url
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONDATE BETWEEN DATE '2020-01-01' AND DATE '2025-12-31'
),

cleaned AS (
  SELECT
    news_time,
    TRIM(REGEXP_REPLACE(raw_title, r'\s+', ' ')) AS title,
    TRIM(REGEXP_REPLACE(theme_keywords, r'\s+', ' ')) AS theme_keywords,
    LOWER(CONCAT(
      ' ', IFNULL(raw_title, ''), ' ',
      IFNULL(v2themes, ''), ' ',
      IFNULL(themes, ''), ' ',
      IFNULL(v2locations, ''), ' ',
      IFNULL(locations, ''), ' ',
      IFNULL(v2organizations, ''), ' ',
      IFNULL(organizations, ''), ' ',
      IFNULL(v2persons, ''), ' ',
      IFNULL(persons, '')
    )) AS haystack,
    url
  FROM base
  WHERE raw_title IS NOT NULL
    AND news_time IS NOT NULL
),

filtered_nonmissing AS (
  SELECT
    news_time,
    title,
    theme_keywords,
    haystack,
    url
  FROM cleaned
  WHERE title IS NOT NULL
    AND title != ''
    AND LENGTH(title) >= 10
),

oil_news AS (
  SELECT
    news_time,
    title,
    theme_keywords,
    url
  FROM filtered_nonmissing
  WHERE
    -- Must contain at least one oil anchor term
    REGEXP_CONTAINS(
      haystack,
      r'\b(oil|crude|petroleum|opec|opec\+|refinery|refineries|pipeline|pipelines|diesel|gasoline|fuel|brent|wti|tanker|tankers|shipment|shipping|terminal|terminals)\b'
    )
    AND
    (
      -- 1) OPEC / OPEC+ supply and production cuts
      REGEXP_CONTAINS(
        haystack,
        r'\b(opec|opec\+)\b'
      )
      OR

      -- 2) Geopolitics / war / sanctions
      REGEXP_CONTAINS(
        haystack,
        r'\b(russia|ukraine|iran|iraq|saudi|saudi arabia|uae|united arab emirates|kuwait|qatar|israel|gaza|hamas|hezbollah|middle east|red sea|houthi|houthis|yemen|syria|libya|sanction|sanctions|war|conflict|strike|drone|missile|attack|invasion)\b'
      )
      OR

      -- 3) Oil transport / shipping / Red Sea / Suez
      REGEXP_CONTAINS(
        haystack,
        r'\b(tanker|tankers|shipment|shipments|shipping|cargo|freight|export|exports|import|imports|red sea|suez|bab el-mandeb|strait of hormuz|hormuz|port|ports)\b'
      )
      OR

      -- 4) Pipeline / refinery / terminal disruptions
      REGEXP_CONTAINS(
        haystack,
        r'\b(pipeline|pipelines|refinery|refineries|terminal|terminals|oilfield|oil field|facility|facilities)\b'
      )
      OR

      -- 5) Demand / consumption / China demand
      REGEXP_CONTAINS(
        haystack,
        r'\b(demand|consumption|consumption growth|throughput|stockpile|stockpiles|inventory|inventories|economic recovery|china|chinese)\b'
      )
    )
),

deduped AS (
  SELECT
    news_time,
    title,
    theme_keywords
  FROM oil_news
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY news_time, title
    ORDER BY theme_keywords
  ) = 1
)

SELECT
  news_time,
  title,
  theme_keywords
FROM deduped
ORDER BY news_time;
