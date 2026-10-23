-- schema.sql — Db2 for i DDL for the credit-card default POC.
-- Placeholder ${SCHEMA} is substituted by load_to_ibmi.py (default CCDEMO).
-- Statements are separated by ';' on their own line end.
--
-- Notes
--  * CREATE SCHEMA (done by load_to_ibmi.py before this file) auto-creates a
--    journal, so INSERTs work even under commit control.
--  * "default.payment.next.month" from the CSV is stored as
--    DEFAULT_PAYMENT_NEXT_MONTH (dots are not valid in SQL identifiers).
--  * All columns are INTEGER/SMALLINT on purpose: they map 1:1 to Arrow ints
--    in MapepireConnector.extract() with no DECIMAL conversion surprises.

CREATE OR REPLACE TABLE ${SCHEMA}.CC_CLIENTS (
    ID                          INTEGER  NOT NULL PRIMARY KEY,
    LIMIT_BAL                   INTEGER  NOT NULL,
    SEX                         SMALLINT NOT NULL,
    EDUCATION                   SMALLINT NOT NULL,
    MARRIAGE                    SMALLINT NOT NULL,
    AGE                         SMALLINT NOT NULL,
    PAY_0                       SMALLINT NOT NULL,
    PAY_2                       SMALLINT NOT NULL,
    PAY_3                       SMALLINT NOT NULL,
    PAY_4                       SMALLINT NOT NULL,
    PAY_5                       SMALLINT NOT NULL,
    PAY_6                       SMALLINT NOT NULL,
    BILL_AMT1                   INTEGER  NOT NULL,
    BILL_AMT2                   INTEGER  NOT NULL,
    BILL_AMT3                   INTEGER  NOT NULL,
    BILL_AMT4                   INTEGER  NOT NULL,
    BILL_AMT5                   INTEGER  NOT NULL,
    BILL_AMT6                   INTEGER  NOT NULL,
    PAY_AMT1                    INTEGER  NOT NULL,
    PAY_AMT2                    INTEGER  NOT NULL,
    PAY_AMT3                    INTEGER  NOT NULL,
    PAY_AMT4                    INTEGER  NOT NULL,
    PAY_AMT5                    INTEGER  NOT NULL,
    PAY_AMT6                    INTEGER  NOT NULL,
    DEFAULT_PAYMENT_NEXT_MONTH  SMALLINT NOT NULL
);

CREATE OR REPLACE TABLE ${SCHEMA}.CC_PREDS (
    ID                    INTEGER  NOT NULL,
    PREDICTION            SMALLINT NOT NULL,
    PROBABILITY_OF_DEFAULT DOUBLE  NOT NULL,
    ACTUAL                SMALLINT,
    MODEL_NAME            VARCHAR(64),
    SCORED_AT             TIMESTAMP NOT NULL DEFAULT CURRENT TIMESTAMP
);
