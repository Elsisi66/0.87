# Phase V Portability Summary

- Generated UTC: 2026-02-22T23:51:34.927345+00:00
- Coins evaluated: SOLUSDT, AVAXUSDT, NEARUSDT, BCHUSDT
- Matrix rows: 16

## Retention vs Baseline

- AVAXUSDT: best=E1 valid=0 delta_vs_E0=0.00079808 cvar_improve=0.182012 maxdd_improve=0.559800
- BCHUSDT: best=E1 valid=0 delta_vs_E0=0.00090029 cvar_improve=0.186479 maxdd_improve=0.430675
- NEARUSDT: best=E1 valid=0 delta_vs_E0=0.00104071 cvar_improve=0.288048 maxdd_improve=0.613271
- SOLUSDT: best=E1 valid=1 delta_vs_E0=0.00089435 cvar_improve=0.190848 maxdd_improve=0.580289

## Best/Runner-Up Per Coin

  symbol  rank exec_choice_id  valid_for_ranking  exec_expectancy_net  delta_vs_e0  cvar_improve_ratio  maxdd_improve_ratio  entry_rate  entries_valid  taker_share  p95_fill_delay_min                         invalid_reason
AVAXUSDT     1             E1                  0            -0.000183     0.000798            0.182012             0.559800    0.972222          175.0     0.114286               15.90                     overall:trades<200
AVAXUSDT     2             E2                  0            -0.000185     0.000796            0.182012             0.559800    0.972222          175.0     0.120000               12.90                     overall:trades<200
 BCHUSDT     1             E1                  0            -0.000335     0.000900            0.186479             0.430675    0.972222          175.0     0.131429               27.90 BCHUSDT:taker_share|overall:trades<200
 BCHUSDT     2             E2                  0            -0.000355     0.000881            0.186479             0.420012    0.972222          175.0     0.148571               24.00 BCHUSDT:taker_share|overall:trades<200
NEARUSDT     1             E1                  0             0.000546     0.001041            0.288048             0.613271    0.911111          164.0     0.054878               20.10 NEARUSDT:entry_rate|overall:trades<200
NEARUSDT     2             E2                  0             0.000542     0.001037            0.252702             0.608771    0.911111          164.0     0.067073               14.55 NEARUSDT:entry_rate|overall:trades<200
 SOLUSDT     1             E1                  1             0.000056     0.000894            0.190848             0.580289    0.986111          355.0     0.081690               15.90                                       
 SOLUSDT     2             E2                  1             0.000056     0.000893            0.182012             0.579504    0.986111          355.0     0.084507               15.00
