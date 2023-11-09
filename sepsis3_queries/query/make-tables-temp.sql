-- As the script is generating many tables, it may take some time.

-- We assume the database and the search path are set correctly.
-- You can set the search path as follows:
-- SET SEARCH_PATH TO public,mimiciii;
-- This will create tables on public and read tables from mimiciii

BEGIN;
-- ----------------------------- --
-- ---------- STAGE 1 ---------- --
-- ----------------------------- --


\i tbls/abx-poe-list.sql
\i tbls/abx-micro-prescription.sql
\i tbls/suspicion-of-infection.sql

-- blood cultures around ICU admission
-- generate cohort
\i tbls/cohort.sql




\i ../mimic-code/concepts/comorbidity/elixhauser_quan.sql

COMMIT;
