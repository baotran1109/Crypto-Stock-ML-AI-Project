# Signal research results

Walk-forward by date, 6 folds over the last 60% of dates, purge gap = target horizon. ROC-AUC 0.5 = random. 'rule' = no-ML baseline.

## crypto / dir5  (test 2020-12-10 -> 2026-11-17)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | rf | 0.522 | 0.481 | 5/6 | +2.33% |
| +price | hgb | 0.516 | 0.496 | 4/6 | +0.69% |
| +volume | hgb | 0.516 | 0.491 | 4/6 | +0.64% |
| all | logreg | 0.513 | 0.492 | 3/6 | +0.54% |
| +cross | logreg | 0.511 | 0.489 | 3/6 | +0.59% |
| all | hgb | 0.509 | 0.450 | 5/6 | +1.13% |
| base | rf | 0.506 | 0.484 | 4/6 | +0.87% |
| base | hgb | 0.505 | 0.482 | 3/6 | +0.93% |
| +cross | hgb | 0.504 | 0.447 | 4/6 | +0.46% |
| base | logreg | 0.499 | 0.467 | 3/6 | +0.80% |
| +volume | logreg | 0.496 | 0.470 | 3/6 | +0.14% |
| - | rule | 0.488 | 0.443 | 3/6 | +1.05% |
| +price | logreg | 0.484 | 0.458 | 2/6 | -0.14% |

## crypto / dir20  (test 2020-12-04 -> 2026-11-02)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | rf | 0.552 | 0.473 | 5/6 | +7.64% |
| +cross | logreg | 0.547 | 0.483 | 4/6 | +7.66% |
| all | hgb | 0.546 | 0.473 | 4/6 | +4.25% |
| all | logreg | 0.546 | 0.489 | 5/6 | +7.18% |
| +cross | hgb | 0.539 | 0.458 | 4/6 | +3.62% |
| +price | hgb | 0.519 | 0.465 | 4/6 | +3.07% |
| +volume | hgb | 0.517 | 0.473 | 3/6 | +2.84% |
| base | rf | 0.513 | 0.433 | 3/6 | +2.69% |
| base | hgb | 0.511 | 0.452 | 3/6 | +2.34% |
| - | rule | 0.506 | 0.437 | 3/6 | +6.12% |
| base | logreg | 0.501 | 0.448 | 3/6 | +3.75% |
| +volume | logreg | 0.471 | 0.424 | 1/6 | -1.57% |
| +price | logreg | 0.465 | 0.423 | 0/6 | -2.34% |

## crypto / excess5  (test 2021-06-30 -> 2026-11-17)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | rf | 0.538 | 0.518 | 6/6 | +0.64% |
| all | hgb | 0.533 | 0.512 | 6/6 | +0.66% |
| +cross | hgb | 0.530 | 0.506 | 6/6 | +0.49% |
| +cross | logreg | 0.529 | 0.511 | 6/6 | +0.55% |
| +price | logreg | 0.527 | 0.509 | 6/6 | +0.75% |
| +volume | logreg | 0.526 | 0.509 | 6/6 | +0.64% |
| +price | hgb | 0.526 | 0.494 | 5/6 | +0.26% |
| all | logreg | 0.525 | 0.509 | 6/6 | +0.56% |
| +volume | hgb | 0.522 | 0.488 | 5/6 | +0.27% |
| base | rf | 0.519 | 0.492 | 5/6 | +0.67% |
| base | logreg | 0.516 | 0.502 | 6/6 | +0.91% |
| base | hgb | 0.516 | 0.475 | 5/6 | +0.55% |
| - | rule | 0.502 | 0.491 | 3/6 | +1.08% |

## crypto / bigmove5  (test 2021-01-09 -> 2026-11-17)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | rf | 0.552 | 0.516 | 6/6 | +2.66% |
| +cross | hgb | 0.550 | 0.534 | 6/6 | +2.44% |
| all | hgb | 0.550 | 0.535 | 6/6 | +2.55% |
| +cross | logreg | 0.545 | 0.501 | 6/6 | +1.59% |
| all | logreg | 0.544 | 0.493 | 5/6 | +1.49% |
| +volume | logreg | 0.544 | 0.511 | 6/6 | +1.19% |
| +volume | hgb | 0.543 | 0.517 | 6/6 | +1.26% |
| +price | logreg | 0.543 | 0.513 | 6/6 | +1.02% |
| +price | hgb | 0.541 | 0.507 | 6/6 | +1.31% |
| base | rf | 0.540 | 0.503 | 6/6 | +2.56% |
| base | hgb | 0.537 | 0.505 | 6/6 | +2.26% |
| base | logreg | 0.528 | 0.491 | 3/6 | +2.55% |
| - | rule | 0.514 | 0.489 | 5/6 | +3.25% |

## stock / dir5  (test 2020-12-15 -> 2026-11-13)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | rf | 0.515 | 0.428 | 3/6 | +0.33% |
| base | logreg | 0.512 | 0.445 | 4/6 | -0.29% |
| +price | hgb | 0.510 | 0.464 | 4/6 | +0.23% |
| +price | logreg | 0.509 | 0.446 | 4/6 | -0.35% |
| base | rf | 0.509 | 0.461 | 4/6 | -0.01% |
| all | hgb | 0.508 | 0.425 | 3/6 | +0.31% |
| +cross | hgb | 0.506 | 0.427 | 3/6 | +0.17% |
| +volume | hgb | 0.505 | 0.464 | 4/6 | +0.09% |
| base | hgb | 0.503 | 0.456 | 3/6 | +0.09% |
| all | logreg | 0.501 | 0.437 | 5/6 | -0.39% |
| +volume | logreg | 0.496 | 0.449 | 4/6 | -0.52% |
| +cross | logreg | 0.495 | 0.436 | 4/6 | -0.51% |
| - | rule | 0.482 | 0.464 | 1/6 | -0.25% |

## stock / dir20  (test 2020-12-07 -> 2026-10-23)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | logreg | 0.536 | 0.430 | 4/6 | -0.34% |
| +volume | logreg | 0.531 | 0.491 | 5/6 | -0.72% |
| +price | logreg | 0.531 | 0.488 | 5/6 | -0.50% |
| base | logreg | 0.530 | 0.454 | 5/6 | -1.00% |
| +cross | logreg | 0.528 | 0.431 | 4/6 | -0.41% |
| all | hgb | 0.506 | 0.412 | 4/6 | -0.81% |
| all | rf | 0.505 | 0.435 | 3/6 | -1.15% |
| +price | hgb | 0.504 | 0.431 | 4/6 | -1.11% |
| +volume | hgb | 0.502 | 0.393 | 4/6 | -1.43% |
| +cross | hgb | 0.498 | 0.412 | 2/6 | -1.06% |
| base | rf | 0.495 | 0.407 | 3/6 | -2.08% |
| base | hgb | 0.492 | 0.404 | 3/6 | -1.45% |
| - | rule | 0.455 | 0.343 | 1/6 | -2.21% |

## stock / excess5  (test 2020-12-15 -> 2026-11-13)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| base | rf | 0.513 | 0.499 | 5/6 | +0.40% |
| all | rf | 0.512 | 0.475 | 3/6 | +0.33% |
| +cross | logreg | 0.509 | 0.476 | 4/6 | +0.23% |
| +price | logreg | 0.507 | 0.467 | 4/6 | +0.34% |
| +volume | logreg | 0.507 | 0.467 | 4/6 | +0.27% |
| all | logreg | 0.506 | 0.461 | 4/6 | +0.19% |
| +price | hgb | 0.505 | 0.493 | 2/6 | +0.15% |
| +volume | hgb | 0.503 | 0.478 | 3/6 | +0.06% |
| base | hgb | 0.502 | 0.463 | 4/6 | +0.07% |
| base | logreg | 0.502 | 0.473 | 3/6 | +0.20% |
| - | rule | 0.500 | 0.463 | 4/6 | +0.24% |
| +cross | hgb | 0.499 | 0.474 | 3/6 | +0.03% |
| all | hgb | 0.499 | 0.475 | 2/6 | -0.03% |

## stock / bigmove5  (test 2021-01-29 -> 2026-11-13)

| features | model | mean AUC | worst fold | folds > 0.5 | top-bottom spread |
|---|---|---|---|---|---|
| all | logreg | 0.582 | 0.551 | 6/6 | +0.87% |
| +cross | logreg | 0.582 | 0.551 | 6/6 | +0.87% |
| all | rf | 0.581 | 0.515 | 6/6 | +1.34% |
| +volume | logreg | 0.580 | 0.556 | 6/6 | +1.29% |
| +price | logreg | 0.578 | 0.549 | 6/6 | +1.22% |
| base | rf | 0.577 | 0.522 | 6/6 | +1.62% |
| +volume | hgb | 0.576 | 0.532 | 6/6 | +1.19% |
| base | logreg | 0.575 | 0.527 | 6/6 | +1.68% |
| base | hgb | 0.575 | 0.529 | 6/6 | +1.45% |
| +price | hgb | 0.573 | 0.533 | 6/6 | +1.18% |
| +cross | hgb | 0.569 | 0.495 | 5/6 | +0.95% |
| all | hgb | 0.568 | 0.494 | 5/6 | +0.94% |
| - | rule | 0.520 | 0.474 | 5/6 | +3.70% |

## crypto rotation (does the excess5 ranking make money after costs?)

Every `hold` bars, hold the top-k coins by predicted P(beat the median coin), equal weight. Walk-forward, period 2021-06-30 -> 2026-11-16. Every configuration tried is listed.

| config | ML total | ML Sharpe | ML max DD | momentum top-k | equal-weight basket | random top-k (median) | ML beats random |
|---|---|---|---|---|---|---|---|
| +cross/logreg top-3 hold 5 cost 0.15% | +11.2% | +0.35 | 86% | +31.2% | +13.8% | -59.2% | 96% |
| +cross/logreg top-3 hold 5 cost 0.30% | -27.6% | +0.23 | 87% | -13.0% | +13.6% | -82.2% | 100% |
| +cross/logreg top-5 hold 5 cost 0.15% | -17.4% | +0.26 | 82% | +151.9% | +13.8% | -39.0% | 79% |
| all/hgb top-3 hold 5 cost 0.15% | -23.4% | +0.23 | 76% | +31.2% | +13.8% | -59.2% | 86% |
| all/hgb top-3 hold 5 cost 0.30% | -54.8% | +0.07 | 80% | -13.0% | +13.6% | -82.2% | 92% |
| all/hgb top-5 hold 5 cost 0.15% | -18.8% | +0.25 | 77% | +151.9% | +13.8% | -39.0% | 78% |
| base/logreg top-3 hold 5 cost 0.15% | -20.7% | +0.26 | 86% | +31.2% | +13.8% | -59.2% | 88% |
| base/logreg top-3 hold 5 cost 0.30% | -56.2% | +0.10 | 88% | -13.0% | +13.6% | -82.2% | 92% |
| base/logreg top-5 hold 5 cost 0.15% | +26.9% | +0.40 | 84% | +151.9% | +13.8% | -39.0% | 96% |
| +cross/logreg top-3 hold 10 cost 0.15% | +31.0% | +0.40 | 82% | +101.6% | +18.3% | -34.0% | 88% |
| +cross/logreg top-5 hold 10 cost 0.15% | +20.9% | +0.38 | 81% | +103.1% | +18.3% | -19.7% | 84% |
| +cross/logreg top-3 hold 20 cost 0.15% | +93.0% | +0.49 | 80% | +31.2% | +38.3% | -3.9% | 88% |
| +cross/logreg top-5 hold 20 cost 0.15% | +86.0% | +0.49 | 77% | +118.0% | +38.3% | +9.9% | 91% |

ML beat the equal-weight basket in 5/13 configs and momentum in 1/13.
