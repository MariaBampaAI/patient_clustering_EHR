-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS echo_data;
CREATE TABLE echo_data AS 
-- This code extracts structured data from echocardiographies
-- You can join it to the text notes using ROW_ID
-- Just note that ROW_ID will differ across versions of MIMIC-III.
SELECT
  ROW_ID,
  subject_id,
  hadm_id,
  chartdate,
  TO_TIMESTAMP(
    chartdate || ' ' || (regexp_matches(ne.text, 'Date/Time: .+? at ([0-9]+:[0-9]{2})'))[1] || ':00',
    'YYYY-MM-DD HH24:MI:SS'
  ) AS charttime,
  (regexp_matches(ne.text, 'Indication: (.*?)\n'))[1] AS Indication,
  CAST((regexp_matches(ne.text, 'Height: \\x28in\\x29 ([0-9]+)'))[1] AS NUMERIC) AS Height,
  CAST((regexp_matches(ne.text, 'Weight \\x28lb\\x29: ([0-9]+)\n'))[1] AS NUMERIC) AS Weight,
  CAST((regexp_matches(ne.text, 'BSA \\x28m2\\x29: ([0-9]+) m2\n'))[1] AS NUMERIC) AS BSA,
  (regexp_matches(ne.text, 'BP \\x28mm Hg\\x29: (.+)\n'))[1] AS BP,
  CAST((regexp_matches(ne.text, 'BP \\x28mm Hg\\x29: ([0-9]+)/[0-9]+?\n'))[1] AS NUMERIC) AS BPSys,
  CAST((regexp_matches(ne.text, 'BP \\x28mm Hg\\x29: [0-9]+/([0-9]+?)\n'))[1] AS NUMERIC) AS BPDias,
  CAST((regexp_matches(ne.text, 'HR \\x28bpm\\x29: ([0-9]+?)\n'))[1] AS NUMERIC) AS HR,
  (regexp_matches(ne.text, 'Status: (.*?)\n'))[1] AS Status,
  (regexp_matches(ne.text, 'Test: (.*?)\n'))[1] AS Test,
  (regexp_matches(ne.text, 'Doppler: (.*?)\n'))[1] AS Doppler,
  (regexp_matches(ne.text, 'Contrast: (.*?)\n'))[1] AS Contrast,
  (regexp_matches(ne.text, 'Technical Quality: (.*?)\n'))[1] AS TechnicalQuality
FROM noteevents ne
WHERE category = 'Echo';
