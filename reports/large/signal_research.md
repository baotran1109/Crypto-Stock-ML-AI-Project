# Signal research results

Walk-forward by date, 6 folds over the last 60% of dates, purge gap = target horizon. ROC-AUC 0.5 = random. 'rule' = no-ML baseline.

## crypto / dir5  (test 2020-12-10 -> 2026-11-17)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| +cross | logreg | 0.540 | 0.496 | 5/6 | +1.94% |
| all | logreg | 0.540 | 0.497 | 5/6 | +1.81% |
| all | hgb | 0.534 | 0.499 | 5/6 | +2.41% |
| +cross | hgb | 0.533 | 0.490 | 5/6 | +2.34% |
| +price | hgb | 0.529 | 0.511 | 6/6 | +1.44% |
| +volume | hgb | 0.527 | 0.511 | 6/6 | +1.11% |
| base | hgb | 0.524 | 0.502 | 6/6 | +1.06% |
| +sentiment | logreg | 0.524 | 0.488 | 4/6 | +0.85% |
| +sentiment | hgb | 0.518 | 0.481 | 5/6 | +0.83% |
| base | logreg | 0.515 | 0.481 | 5/6 | +0.81% |
| +volume | logreg | 0.515 | 0.481 | 4/6 | +0.56% |
| +price | logreg | 0.514 | 0.480 | 4/6 | +0.48% |
| - | rule | 0.488 | 0.444 | 2/6 | +1.09% |

## crypto / dir20  (test 2020-12-04 -> 2026-11-02)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | logreg | 0.586 | 0.511 | 6/6 | +11.87% |
| +cross | logreg | 0.583 | 0.505 | 6/6 | +12.55% |
| all | hgb | 0.563 | 0.515 | 6/6 | +8.69% |
| +sentiment | logreg | 0.557 | 0.492 | 5/6 | +5.22% |
| +cross | hgb | 0.555 | 0.481 | 5/6 | +7.66% |
| +sentiment | hgb | 0.535 | 0.470 | 4/6 | +4.69% |
| +volume | hgb | 0.523 | 0.455 | 5/6 | +3.47% |
| +price | hgb | 0.521 | 0.464 | 4/6 | +2.83% |
| base | logreg | 0.520 | 0.468 | 5/6 | +3.33% |
| base | hgb | 0.514 | 0.447 | 5/6 | +3.02% |
| - | rule | 0.507 | 0.435 | 3/6 | +6.57% |
| +volume | logreg | 0.503 | 0.467 | 4/6 | +0.47% |
| +price | logreg | 0.500 | 0.467 | 3/6 | +0.14% |

## crypto / excess5  (test 2021-06-30 -> 2026-11-17)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| +sentiment | hgb | 0.533 | 0.520 | 6/6 | +0.38% |
| +cross | hgb | 0.531 | 0.518 | 6/6 | +0.40% |
| +volume | hgb | 0.531 | 0.523 | 6/6 | +0.42% |
| all | hgb | 0.531 | 0.518 | 6/6 | +0.52% |
| +price | hgb | 0.528 | 0.515 | 6/6 | +0.57% |
| +price | logreg | 0.527 | 0.508 | 6/6 | +0.55% |
| +cross | logreg | 0.527 | 0.509 | 6/6 | +0.35% |
| +sentiment | logreg | 0.526 | 0.510 | 6/6 | +0.33% |
| +volume | logreg | 0.525 | 0.508 | 6/6 | +0.40% |
| all | logreg | 0.523 | 0.503 | 6/6 | +0.30% |
| base | hgb | 0.521 | 0.512 | 6/6 | +0.64% |
| base | logreg | 0.520 | 0.506 | 6/6 | +0.53% |
| - | rule | 0.495 | 0.479 | 2/6 | +1.36% |

## crypto / bigmove5  (test 2021-01-09 -> 2026-11-17)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| +sentiment | logreg | 0.565 | 0.506 | 6/6 | +2.93% |
| +cross | hgb | 0.565 | 0.550 | 6/6 | +3.34% |
| +price | hgb | 0.562 | 0.529 | 6/6 | +2.82% |
| +cross | logreg | 0.562 | 0.523 | 6/6 | +2.53% |
| all | hgb | 0.562 | 0.548 | 6/6 | +3.16% |
| +volume | logreg | 0.561 | 0.528 | 6/6 | +2.38% |
| +volume | hgb | 0.561 | 0.526 | 6/6 | +2.75% |
| all | logreg | 0.561 | 0.512 | 6/6 | +2.59% |
| +price | logreg | 0.561 | 0.527 | 6/6 | +2.34% |
| +sentiment | hgb | 0.557 | 0.504 | 6/6 | +3.34% |
| base | hgb | 0.553 | 0.526 | 6/6 | +3.21% |
| base | logreg | 0.550 | 0.506 | 6/6 | +3.52% |
| - | rule | 0.530 | 0.515 | 6/6 | +3.72% |

## stock / dir5  (test 2020-12-15 -> 2026-11-13)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | logreg | 0.516 | 0.468 | 4/6 | +0.31% |
| +macro | logreg | 0.514 | 0.457 | 5/6 | +0.21% |
| +events | logreg | 0.512 | 0.459 | 5/6 | +0.22% |
| +cross | logreg | 0.512 | 0.465 | 5/6 | +0.15% |
| +cross | hgb | 0.512 | 0.475 | 5/6 | +0.23% |
| +events | hgb | 0.510 | 0.493 | 3/6 | +0.32% |
| all | hgb | 0.510 | 0.475 | 5/6 | +0.20% |
| +macro | hgb | 0.506 | 0.486 | 3/6 | +0.27% |
| +price | hgb | 0.506 | 0.496 | 3/6 | +0.21% |
| +volume | hgb | 0.506 | 0.492 | 4/6 | +0.23% |
| +price | logreg | 0.504 | 0.495 | 3/6 | +0.22% |
| base | logreg | 0.504 | 0.498 | 4/6 | +0.22% |
| base | hgb | 0.503 | 0.488 | 4/6 | +0.15% |
| +volume | logreg | 0.501 | 0.486 | 4/6 | +0.07% |
| - | rule_eps | 0.500 | 0.488 | 3/6 | +0.07% |
| - | rule | 0.497 | 0.487 | 2/6 | -0.14% |

## stock / dir20  (test 2020-12-07 -> 2026-10-23)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| +macro | logreg | 0.524 | 0.479 | 3/6 | +1.33% |
| +events | logreg | 0.523 | 0.474 | 3/6 | +1.37% |
| base | logreg | 0.518 | 0.505 | 6/6 | +1.41% |
| +price | logreg | 0.515 | 0.489 | 4/6 | +1.30% |
| +volume | logreg | 0.512 | 0.483 | 4/6 | +1.14% |
| all | logreg | 0.511 | 0.466 | 4/6 | +0.95% |
| +price | hgb | 0.510 | 0.469 | 4/6 | +0.74% |
| +volume | hgb | 0.508 | 0.467 | 4/6 | +0.74% |
| +cross | logreg | 0.505 | 0.452 | 4/6 | +0.60% |
| base | hgb | 0.504 | 0.477 | 3/6 | +0.53% |
| - | rule_eps | 0.503 | 0.486 | 3/6 | +0.28% |
| +macro | hgb | 0.500 | 0.472 | 3/6 | +0.19% |
| +events | hgb | 0.499 | 0.485 | 3/6 | +0.35% |
| all | hgb | 0.490 | 0.443 | 2/6 | -0.16% |
| +cross | hgb | 0.485 | 0.428 | 2/6 | -0.28% |
| - | rule | 0.475 | 0.445 | 1/6 | -1.33% |

## stock / excess5  (test 2020-12-15 -> 2026-11-13)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| +macro | hgb | 0.510 | 0.504 | 6/6 | +0.19% |
| all | hgb | 0.508 | 0.490 | 5/6 | +0.17% |
| +cross | hgb | 0.508 | 0.489 | 5/6 | +0.17% |
| +macro | logreg | 0.508 | 0.491 | 4/6 | +0.12% |
| +cross | logreg | 0.507 | 0.491 | 4/6 | +0.12% |
| +events | logreg | 0.507 | 0.499 | 5/6 | +0.06% |
| +events | hgb | 0.507 | 0.499 | 5/6 | +0.13% |
| all | logreg | 0.506 | 0.491 | 3/6 | +0.14% |
| +price | logreg | 0.505 | 0.491 | 4/6 | +0.07% |
| +volume | logreg | 0.504 | 0.490 | 4/6 | +0.07% |
| base | hgb | 0.503 | 0.490 | 3/6 | +0.11% |
| base | logreg | 0.503 | 0.487 | 4/6 | +0.03% |
| +volume | hgb | 0.503 | 0.486 | 3/6 | +0.06% |
| +price | hgb | 0.502 | 0.485 | 3/6 | +0.07% |
| - | rule | 0.502 | 0.497 | 4/6 | +0.03% |
| - | rule_eps | 0.502 | 0.488 | 3/6 | +0.02% |

## stock / bigmove5  (test 2021-01-29 -> 2026-11-13)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| +events | hgb | 0.558 | 0.523 | 6/6 | +0.94% |
| +events | logreg | 0.557 | 0.533 | 6/6 | +1.06% |
| all | logreg | 0.552 | 0.519 | 6/6 | +0.88% |
| +cross | logreg | 0.552 | 0.520 | 6/6 | +0.87% |
| +volume | logreg | 0.548 | 0.518 | 6/6 | +1.00% |
| +volume | hgb | 0.547 | 0.520 | 6/6 | +0.93% |
| +macro | logreg | 0.546 | 0.517 | 6/6 | +0.74% |
| +macro | hgb | 0.544 | 0.499 | 5/6 | +0.51% |
| +price | hgb | 0.543 | 0.518 | 6/6 | +0.84% |
| +price | logreg | 0.543 | 0.514 | 6/6 | +0.91% |
| base | logreg | 0.541 | 0.515 | 6/6 | +1.11% |
| base | hgb | 0.541 | 0.515 | 6/6 | +1.15% |
| all | hgb | 0.541 | 0.519 | 6/6 | +0.54% |
| +cross | hgb | 0.541 | 0.519 | 6/6 | +0.54% |
| - | rule | 0.509 | 0.494 | 4/6 | +2.08% |
| - | rule_eps | 0.493 | 0.473 | 1/6 | +0.19% |

## crypto rotation (does the excess5 ranking make money after costs?)

Every `hold` bars, hold the top-k symbols by predicted P(beat the median), equal weight. Walk-forward, period 2021-06-30 -> 2026-11-16. Every configuration tried is listed.

| config | ML total | ML Sharpe | ML max DD | momentum top-k | equal-weight basket | random top-k (median) | ML beats random |
|---|---|---|---|---|---|---|---|
| +cross/logreg top-3 hold 5 cost 0.15% | +621.4% | +0.82 | 73% | +592.1% | -45.1% | -89.6% | 100% |
| +cross/logreg top-3 hold 5 cost 0.30% | +328.8% | +0.69 | 77% | +317.8% | -45.2% | -96.5% | 100% |
| +cross/logreg top-5 hold 5 cost 0.15% | +306.4% | +0.66 | 82% | +170.7% | -45.1% | -84.4% | 100% |
| all/hgb top-3 hold 5 cost 0.15% | +495.2% | +0.77 | 71% | +592.1% | -45.1% | -89.6% | 100% |
| all/hgb top-3 hold 5 cost 0.30% | +233.8% | +0.63 | 73% | +317.8% | -45.2% | -96.5% | 100% |
| all/hgb top-5 hold 5 cost 0.15% | +358.2% | +0.74 | 72% | +170.7% | -45.1% | -84.4% | 100% |
| base/logreg top-3 hold 5 cost 0.15% | +32.5% | +0.41 | 86% | +592.1% | -45.1% | -89.6% | 99% |
| base/logreg top-3 hold 5 cost 0.30% | -35.3% | +0.24 | 90% | +317.8% | -45.2% | -96.5% | 100% |
| base/logreg top-5 hold 5 cost 0.15% | +76.9% | +0.48 | 87% | +170.7% | -45.1% | -84.4% | 100% |
| +cross/logreg top-3 hold 10 cost 0.15% | +424.0% | +0.70 | 84% | +156.9% | -44.0% | -80.5% | 100% |
| +cross/logreg top-5 hold 10 cost 0.15% | +194.9% | +0.59 | 84% | +150.7% | -44.0% | -75.4% | 100% |
| +cross/logreg top-3 hold 20 cost 0.15% | +150.3% | +0.49 | 84% | +107.1% | -32.7% | -66.9% | 98% |
| +cross/logreg top-5 hold 20 cost 0.15% | +81.4% | +0.43 | 81% | +34.3% | -32.7% | -54.9% | 98% |

ML beat the equal-weight basket in 13/13 configs and momentum in 8/13.

## stock rotation (does the excess5 ranking make money after costs?)

Every `hold` bars, hold the top-k symbols by predicted P(beat the median), equal weight. Walk-forward, period 2020-12-15 -> 2026-11-13. Every configuration tried is listed.

| config | ML total | ML Sharpe | ML max DD | momentum top-k | equal-weight basket | random top-k (median) | ML beats random |
|---|---|---|---|---|---|---|---|
| +cross/logreg top-3 hold 5 cost 0.15% | -22.0% | +0.05 | 67% | +249.3% | +99.5% | -20.8% | 49% |
| +cross/logreg top-3 hold 5 cost 0.30% | -60.5% | -0.28 | 75% | +115.1% | +99.2% | -66.7% | 66% |
| +cross/logreg top-5 hold 5 cost 0.15% | +5.3% | +0.18 | 57% | +227.7% | +99.5% | -21.8% | 84% |
| all/hgb top-3 hold 5 cost 0.15% | +43.4% | +0.35 | 41% | +249.3% | +99.5% | -20.8% | 93% |
| all/hgb top-3 hold 5 cost 0.30% | -25.0% | +0.01 | 55% | +115.1% | +99.2% | -66.7% | 100% |
| all/hgb top-5 hold 5 cost 0.15% | +65.9% | +0.45 | 37% | +227.7% | +99.5% | -21.8% | 99% |
| base/logreg top-3 hold 5 cost 0.15% | -32.1% | -0.04 | 70% | +249.3% | +99.5% | -20.8% | 34% |
| base/logreg top-3 hold 5 cost 0.30% | -68.3% | -0.44 | 77% | +115.1% | +99.2% | -66.7% | 44% |
| base/logreg top-5 hold 5 cost 0.15% | -9.3% | +0.09 | 60% | +227.7% | +99.5% | -21.8% | 70% |
| +cross/logreg top-3 hold 10 cost 0.15% | +36.2% | +0.32 | 58% | +290.0% | +99.8% | +20.1% | 60% |
| +cross/logreg top-5 hold 10 cost 0.15% | +47.4% | +0.37 | 54% | +205.3% | +99.8% | +23.1% | 76% |
| +cross/logreg top-3 hold 20 cost 0.15% | +250.8% | +0.74 | 40% | +243.8% | +111.1% | +54.3% | 100% |
| +cross/logreg top-5 hold 20 cost 0.15% | +138.2% | +0.62 | 38% | +256.2% | +111.1% | +72.6% | 90% |

ML beat the equal-weight basket in 2/13 configs and momentum in 1/13.
