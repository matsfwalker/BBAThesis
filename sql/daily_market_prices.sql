SELECT
    a.gvkey,
    a.datadate AS date,
    a.prccd AS close,
    a.cshoc AS sharesoutstanding,
    b.exchgdesc AS exchange
FROM (
    SELECT
        gvkey,
        datadate,
        prccd,
        cshoc,
        exchg
    FROM comp.secd
    WHERE
        datadate BETWEEN %s AND %s  -- Filter on dates specified in the query parameters
        AND tpci = '0'              -- Common shares only (filter out ETFs and similar)
        AND prccd IS NOT NULL
        AND cshoc IS NOT NULL
        AND cshoc > 0
) AS a
JOIN comp.r_ex_codes AS b
    ON a.exchg = b.exchgcd;         -- Get the name of the main exchange of the stock