-- As the script is generating many tables, it may take some time.

-- We extract the vasopressor doses for all patients in the ICU
-- Needs later preprocessing to bring the tables together and select the appropriate cohort

BEGIN;


\i ../mimic-code/concepts/durations/weight_durations.sql

\i ../mimic-code/concepts/durations/dobutamine_dose.sql
\i ../mimic-code/concepts/durations/dopamine_dose.sql
\i ../mimic-code/concepts/durations/vasopressin_dose.sql
\i ../mimic-code/concepts/durations/phenylephrine_dose.sql
\i ../mimic-code/concepts/durations/epinephrine_dose.sql
\i ../mimic-code/concepts/durations/norepinephrine_dose.sql





COMMIT;