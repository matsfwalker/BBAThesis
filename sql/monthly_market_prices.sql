SELECT
    a.gvkey,
    a.datadate AS date,
    a.prccm AS close,
    a.trt1m AS return_factor,
    a.cshom AS sharesoutstanding,
    b.exchgcd AS exchange
FROM (
    SELECT
        gvkey,
        datadate,
        prccm,
        cshom,
        exchg,
        trt1m
    FROM comp.secm
    WHERE
        datadate BETWEEN %s AND %s  -- Filter on dates specified in the query parameters
        AND tpci = '0'              -- Common shares only (filter out ETFs and similar)
        AND prccm IS NOT NULL
        AND cshom IS NOT NULL
        AND cshom > 0
) AS a
JOIN comp.r_ex_codes AS b
    ON a.exchg = b.exchgcd;         -- Get the name of the main exchange of the stock