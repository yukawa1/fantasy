# FANTASY calibration: 1500 spectra, 5700 fits

## Fit quality per group and variant

| group | variant | n | failed | rchi2 | rchi2_90 | nfev | time | at_limit | free |
|---|---|---|---|---|---|---|---|---|---|
| comp | composite | 100 | 0 | 1.01 | 1.3 | 15 | 10.2 | 1.16 | 85 |
| comp | composite no outflow | 100 | 0 | 1.03 | 1.44 | 8 | 3.42 | 0.05 | 76 |
| liner | liner | 150 | 0 | 0.99 | 1.23 | 7 | 2.78 | 0.01 | 74 |
| liner | liner+broad | 150 | 0 | 0.97 | 1.19 | 9 | 3.33 | 0.97 | 78 |
| liner | liner+outflow | 150 | 0 | 0.97 | 1.16 | 11 | 8.46 | 0.99 | 83 |
| passive | passive | 100 | 0 | 0.97 | 1.13 | 8 | 1.99 | 0.86 | 53 |
| sf | starforming | 150 | 0 | 1.05 | 1.29 | 18 | 6.82 | 0.88 | 66 |
| sf | starforming no outflow | 150 | 0 | 1.06 | 1.32 | 8 | 2.24 | 0.01 | 58 |
| sy2 | seyfert2 | 150 | 0 | 1.02 | 1.22 | 13 | 9.51 | 0.6 | 85 |
| sy2 | seyfert2 no outflow | 150 | 0 | 1.07 | 1.59 | 8 | 3.21 | 0.01 | 76 |
| sy2 | seyfert2 one system | 150 | 0 | 1.03 | 1.28 | 13 | 8.37 | 0.62 | 83 |
| t1_civ | civ | 150 | 0 | 1.57 | 6.2 | 12 | 0.08 | 1.33 | 9 |
| t1_civ | civ 2 broad | 150 | 0 | 1.29 | 5.06 | 21 | 0.28 | 1.11 | 12 |
| t1_hb | hbeta | 200 | 0 | 1 | 1.52 | 22 | 3.23 | 0.96 | 35 |
| t1_hb | hbeta 1 broad | 200 | 0 | 1.07 | 1.67 | 19 | 1.13 | 1.38 | 31 |
| t1_hb | hbeta 3 broad | 200 | 0 | 0.99 | 1.49 | 28 | 3.93 | 1.56 | 39 |
| t1_hb | hbeta+outflow | 200 | 0 | 0.95 | 1.34 | 27 | 3.86 | 1.2 | 38 |
| t1_low | halpha | 300 | 0 | 1.23 | 1.81 | 18.5 | 0.36 | 0.4 | 17 |
| t1_low | halpha 1 broad | 300 | 0 | 1.53 | 2.9 | 12 | 0.11 | 0.16 | 14 |
| t1_low | halpha+outflow | 300 | 0 | 1.11 | 1.58 | 26 | 0.53 | 0.8 | 21 |
| t1_low | hbeta | 300 | 0 | 1.22 | 1.85 | 25 | 3.66 | 1.64 | 35 |
| t1_low | hbeta 1 broad | 300 | 0 | 1.31 | 2.11 | 19 | 1.24 | 1.82 | 31 |
| t1_low | hbeta 3 broad | 300 | 0 | 1.21 | 1.82 | 28 | 4.39 | 2.39 | 39 |
| t1_low | hbeta+outflow | 300 | 0 | 1.13 | 1.52 | 27 | 4.36 | 1.85 | 38 |
| t1_low | optical | 300 | 0 | 1.36 | 2.09 | 30 | 18.5 | 3.12 | 67 |
| t1_low | optical+outflow | 300 | 0 | 1.26 | 1.88 | 34 | 21.7 | 3.31 | 71 |
| t1_mgii | mgii | 200 | 0 | 0.87 | 1.3 | 23 | 0.24 | 1.17 | 17 |
| t1_mgii | mgii 2 broad | 200 | 0 | 0.85 | 1.25 | 35.5 | 0.72 | 1.59 | 20 |

## Parameters at a limit in at least 10% of the fits

| group | variant | param | fraction |
|---|---|---|---|
| liner | liner+outflow | outflow.fwhm | 0.75 |
| t1_civ | civ | heii.shift | 0.74 |
| comp | composite | outflow.fwhm | 0.73 |
| t1_low | optical+outflow | broad_he.fwhm | 0.73 |
| t1_low | optical | broad_he.fwhm | 0.72 |
| sf | starforming | outflow.fwhm | 0.65 |
| t1_mgii | mgii | narrow.fwhm | 0.64 |
| passive | passive | narrow.fwhm | 0.61 |
| liner | liner+broad | broad.fwhm | 0.59 |
| sy2 | seyfert2 one system | outflow.fwhm | 0.57 |
| t1_civ | civ 2 broad | heii.shift | 0.55 |
| t1_civ | civ | heii.fwhm | 0.53 |
| sy2 | seyfert2 | outflow.fwhm | 0.5 |
| t1_low | optical+outflow | feii.shift | 0.49 |
| t1_mgii | mgii 2 broad | continuum.index | 0.46 |
| t1_mgii | mgii 2 broad | narrow.fwhm | 0.46 |
| t1_low | optical | feii.shift | 0.45 |
| t1_hb | hbeta 1 broad | heii.shift | 0.45 |
| t1_low | hbeta 1 broad | feii.shift | 0.39 |
| t1_low | hbeta | continuum.index | 0.38 |
| liner | liner+broad | broad.shift | 0.38 |
| t1_hb | hbeta 1 broad | heii.fwhm | 0.37 |
| t1_low | hbeta 3 broad | continuum.index | 0.36 |
| t1_low | optical | broad_ha2.fwhm | 0.35 |
| t1_low | hbeta 1 broad | heii.shift | 0.35 |
| t1_low | halpha+outflow | broad2.shift | 0.34 |
| t1_low | hbeta+outflow | continuum.index | 0.34 |
| comp | composite | outflow.shift | 0.34 |
| t1_low | optical | feii.fwhm | 0.34 |
| t1_low | optical+outflow | broad_ha2.fwhm | 0.33 |
| t1_low | optical | broad2.shift | 0.33 |
| t1_low | optical+outflow | feii.fwhm | 0.32 |
| t1_low | hbeta+outflow | feii.shift | 0.32 |
| t1_mgii | mgii | continuum.index | 0.32 |
| t1_low | hbeta 3 broad | feii.shift | 0.31 |
| t1_low | hbeta 3 broad | broad3.shift | 0.31 |
| t1_low | hbeta 1 broad | heii.fwhm | 0.31 |
| t1_low | hbeta | heii.shift | 0.28 |
| t1_low | optical+outflow | continuum.index_blue | 0.28 |
| t1_low | hbeta+outflow | heii.shift | 0.28 |
| t1_low | optical | continuum.index_blue | 0.26 |
| t1_low | hbeta 1 broad | broad.fwhm | 0.25 |
| t1_low | halpha+outflow | outflow.fwhm | 0.25 |
| t1_low | hbeta | feii.shift | 0.24 |
| t1_low | hbeta 3 broad | broad2.fwhm | 0.24 |
| t1_civ | civ 2 broad | heii.fwhm | 0.24 |
| t1_hb | hbeta 3 broad | broad3.shift | 0.24 |
| liner | liner+outflow | outflow.shift | 0.23 |
| t1_low | hbeta 3 broad | heii.shift | 0.23 |
| t1_mgii | mgii 2 broad | narrow.shift | 0.22 |
| t1_low | hbeta | broad2.shift | 0.22 |
| t1_hb | hbeta | heii.shift | 0.22 |
| passive | passive | narrow.shift | 0.22 |
| t1_low | optical+outflow | broad2.shift | 0.22 |
| sf | starforming | outflow.shift | 0.21 |
| t1_hb | hbeta 3 broad | broad2.fwhm | 0.21 |
| t1_hb | hbeta+outflow | heii.shift | 0.21 |
| t1_hb | hbeta 3 broad | broad.fwhm | 0.2 |
| t1_low | hbeta 3 broad | feii.fwhm | 0.2 |
| t1_low | hbeta+outflow | heii.fwhm | 0.19 |
| t1_low | halpha | broad2.shift | 0.19 |
| t1_low | hbeta 3 broad | broad.fwhm | 0.18 |
| t1_low | hbeta 3 broad | broad3.fwhm | 0.18 |
| t1_low | hbeta | heii.fwhm | 0.18 |
| t1_low | optical+outflow | broad.fwhm | 0.18 |
| t1_low | hbeta+outflow | outflow.fwhm | 0.17 |
| t1_low | hbeta 1 broad | feii.fwhm | 0.17 |
| t1_mgii | mgii 2 broad | broad2.fwhm | 0.17 |
| t1_low | hbeta 3 broad | broad2.shift | 0.17 |
| t1_hb | hbeta | broad2.shift | 0.17 |
| t1_low | hbeta 1 broad | broad.shift | 0.17 |
| t1_mgii | mgii | narrow.shift | 0.16 |
| t1_hb | hbeta 1 broad | narrow.fwhm | 0.16 |
| t1_hb | hbeta+outflow | heii.fwhm | 0.16 |
| t1_low | hbeta 3 broad | heii.fwhm | 0.16 |
| t1_low | optical | broad.fwhm | 0.16 |
| t1_low | optical | broad.shift | 0.16 |
| t1_low | optical+outflow | outflow.fwhm | 0.16 |
| t1_low | hbeta 1 broad | continuum.index | 0.16 |
| t1_hb | hbeta 3 broad | heii.shift | 0.16 |
| t1_hb | hbeta 1 broad | feii.fwhm | 0.16 |
| t1_low | optical+outflow | broad.shift | 0.15 |
| t1_low | hbeta+outflow | broad2.shift | 0.15 |
| t1_hb | hbeta+outflow | outflow.fwhm | 0.14 |
| t1_low | hbeta+outflow | broad2.fwhm | 0.14 |
| t1_hb | hbeta 3 broad | heii.fwhm | 0.14 |
| t1_civ | civ 2 broad | broad2.shift | 0.14 |
| t1_hb | hbeta 1 broad | feii.shift | 0.14 |
| t1_low | optical+outflow | broad_ha2.shift | 0.13 |
| t1_low | hbeta | feii.fwhm | 0.13 |
| t1_hb | hbeta 3 broad | broad3.fwhm | 0.13 |
| t1_hb | hbeta+outflow | narrow.fwhm | 0.12 |
| t1_hb | hbeta 3 broad | continuum.index | 0.12 |
| t1_low | optical+outflow | broad_he.shift | 0.12 |
| t1_low | optical | broad_ha2.shift | 0.11 |
| t1_low | halpha | broad2.fwhm | 0.11 |
| t1_low | optical | broad_he.shift | 0.11 |
| t1_hb | hbeta | continuum.index | 0.11 |
| t1_hb | hbeta 3 broad | broad2.shift | 0.11 |
| t1_hb | hbeta+outflow | feii.shift | 0.11 |
| t1_civ | civ 2 broad | broad2.fwhm | 0.11 |
| t1_hb | hbeta | heii.fwhm | 0.1 |
| t1_low | hbeta | broad2.fwhm | 0.1 |
| t1_low | halpha 1 broad | narrow.fwhm | 0.1 |

## BIC preference (same group and wavelength range)

| group | variant | n | wins | median_dbic |
|---|---|---|---|---|
| comp | composite | 100 | 0.24 | 40.2 |
| comp | composite no outflow | 100 | 0.76 | 0 |
| liner | liner | 150 | 0.39 | 11.7 |
| liner | liner+broad | 150 | 0.44 | 3.68 |
| liner | liner+outflow | 150 | 0.17 | 40 |
| sf | starforming | 150 | 0.16 | 42.6 |
| sf | starforming no outflow | 150 | 0.84 | 0 |
| sy2 | seyfert2 | 150 | 0.37 | 8.43 |
| sy2 | seyfert2 no outflow | 150 | 0.42 | 25.6 |
| sy2 | seyfert2 one system | 150 | 0.21 | 31.7 |
| t1_civ | civ | 150 | 0.05 | 148 |
| t1_civ | civ 2 broad | 150 | 0.95 | 0 |
| t1_hb | hbeta | 200 | 0.16 | 21 |
| t1_hb | hbeta 1 broad | 200 | 0.2 | 46.2 |
| t1_hb | hbeta 3 broad | 200 | 0.04 | 36.4 |
| t1_hb | hbeta+outflow | 200 | 0.6 | 0 |
| t1_low | halpha | 300 | 0.28 | 15.6 |
| t1_low | halpha 1 broad | 300 | 0.11 | 105 |
| t1_low | halpha+outflow | 300 | 0.61 | 0 |
| t1_low | hbeta | 300 | 0.13 | 59.5 |
| t1_low | hbeta 1 broad | 300 | 0.05 | 162 |
| t1_low | hbeta 3 broad | 300 | 0.11 | 50.7 |
| t1_low | hbeta+outflow | 300 | 0.7 | 0 |
| t1_low | optical | 300 | 0.13 | 88.9 |
| t1_low | optical+outflow | 300 | 0.87 | 0 |
| t1_mgii | mgii | 200 | 0.8 | 0 |
| t1_mgii | mgii 2 broad | 200 | 0.2 | 11.5 |

## Nonlinear parameters (calibration half): distribution, current and suggested limits

| group | variant | param | n | p0.5 | median | p99.5 | min | max | at_min | at_max | suggest_value | suggest_min | suggest_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comp | composite | host.ebv | 76 | 0.012 | 0.241 | 0.462 | 0 | 1.5 | 0.013 | 0 | 0.241 | -0.079 | 0.56 |
| comp | composite | host.sigma | 76 | 41.7 | 124 | 254 | 30 | 450 | 0 | 0 | 124 | -0.92 | 300 |
| comp | composite | host.velocity | 76 | -53.3 | -0.445 | 43 | -600 | 600 | 0 | 0 | -0.445 | -73 | 63 |
| comp | composite | narrow.fwhm | 76 | 115 | 249 | 619 | 50 | 1.2e+03 | 0 | 0 | 249 | 13 | 720 |
| comp | composite | narrow.shift | 76 | -123 | 0.233 | 22.1 | -600 | 600 | 0 | 0 | 0.233 | -160 | 52 |
| comp | composite | narrow_hi.fwhm | 76 | 50 | 274 | 745 | 50 | 1.5e+03 | 0.053 | 0 | 274 | -680 | 890 |
| comp | composite | narrow_hi.shift | 76 | -465 | -7.21 | 491 | -600 | 600 | 0.013 | 0.013 | -7.21 | -660 | 690 |
| comp | composite | outflow.fwhm | 76 | 500 | 2.48e+03 | 2.5e+03 | 500 | 2.5e+03 | 0.25 | 0.5 | 2.48e+03 | -500 | 3.5e+03 |
| comp | composite | outflow.shift | 76 | -1.5e+03 | -17.2 | 500 | -1.5e+03 | 500 | 0.132 | 0.197 | -17.2 | -2.5e+03 | 1.5e+03 |
| comp | composite no outflow | host.ebv | 76 | 0.012 | 0.227 | 0.461 | 0 | 1.5 | 0.013 | 0 | 0.227 | -0.078 | 0.56 |
| comp | composite no outflow | host.sigma | 76 | 61.2 | 126 | 255 | 30 | 450 | 0 | 0 | 126 | 22 | 300 |
| comp | composite no outflow | host.velocity | 76 | -56.2 | -0.687 | 53.4 | -600 | 600 | 0 | 0 | -0.687 | -79 | 76 |
| comp | composite no outflow | narrow.fwhm | 76 | 142 | 289 | 644 | 50 | 1.2e+03 | 0 | 0 | 289 | 41 | 750 |
| comp | composite no outflow | narrow.shift | 76 | -29.4 | 0.129 | 23.5 | -600 | 600 | 0 | 0 | 0.129 | -40 | 35 |
| comp | composite no outflow | narrow_hi.fwhm | 76 | 89.2 | 295 | 1.5e+03 | 50 | 1.5e+03 | 0 | 0.026 | 295 | -200 | 2.3e+03 |
| comp | composite no outflow | narrow_hi.shift | 76 | -424 | -7.32 | 93 | -600 | 600 | 0.013 | 0 | -7.32 | -530 | 200 |
| liner | liner | host.ebv | 106 | 0.01 | 0.141 | 0.516 | 0 | 1.5 | 0 | 0 | 0.141 | -0.092 | 0.62 |
| liner | liner | host.sigma | 106 | 84 | 158 | 294 | 30 | 450 | 0 | 0 | 158 | 41 | 340 |
| liner | liner | host.velocity | 106 | -34 | 4.26 | 32.8 | -600 | 600 | 0 | 0 | 4.26 | -48 | 47 |
| liner | liner | narrow.fwhm | 106 | 182 | 409 | 1.12e+03 | 50 | 1.5e+03 | 0 | 0.009 | 409 | -5.5 | 1.4e+03 |
| liner | liner | narrow.shift | 106 | -30.2 | 4.59 | 49.2 | -600 | 600 | 0 | 0 | 4.59 | -47 | 66 |
| liner | liner+broad | broad.fwhm | 106 | 1.2e+03 | 8e+03 | 8e+03 | 1.2e+03 | 8e+03 | 0.047 | 0.538 | 8e+03 | -2.2e+03 | 1.2e+04 |
| liner | liner+broad | broad.shift | 106 | -2e+03 | 1.3e+03 | 2e+03 | -2e+03 | 2e+03 | 0.019 | 0.406 | 1.3e+03 | -2.8e+03 | 4e+03 |
| liner | liner+broad | host.ebv | 106 | 0.011 | 0.145 | 0.541 | 0 | 1.5 | 0 | 0 | 0.145 | -0.095 | 0.65 |
| liner | liner+broad | host.sigma | 106 | 79 | 158 | 294 | 30 | 450 | 0 | 0 | 158 | 35 | 340 |
| liner | liner+broad | host.velocity | 106 | -33.6 | 4.37 | 33.3 | -600 | 600 | 0 | 0 | 4.37 | -47 | 47 |
| liner | liner+broad | narrow.fwhm | 106 | 157 | 374 | 1.11e+03 | 50 | 1.5e+03 | 0 | 0.009 | 374 | -34 | 1.3e+03 |
| liner | liner+broad | narrow.shift | 106 | -28.2 | 4.54 | 49.5 | -600 | 600 | 0 | 0 | 4.54 | -44 | 66 |
| liner | liner+outflow | host.ebv | 106 | 0.009 | 0.143 | 0.537 | 0 | 1.5 | 0 | 0 | 0.143 | -0.096 | 0.65 |
| liner | liner+outflow | host.sigma | 106 | 74.6 | 158 | 296 | 30 | 450 | 0 | 0 | 158 | 30 | 350 |
| liner | liner+outflow | host.velocity | 106 | -31.6 | 4.43 | 34.1 | -600 | 600 | 0 | 0 | 4.43 | -45 | 48 |
| liner | liner+outflow | narrow.fwhm | 106 | 138 | 338 | 1.12e+03 | 50 | 1.5e+03 | 0 | 0.009 | 338 | -60 | 1.4e+03 |
| liner | liner+outflow | narrow.shift | 106 | -94.4 | 5.61 | 72.7 | -600 | 600 | 0 | 0 | 5.61 | -130 | 110 |
| liner | liner+outflow | outflow.fwhm | 106 | 500 | 2.5e+03 | 2.5e+03 | 500 | 2.5e+03 | 0.085 | 0.66 | 2.5e+03 | -500 | 3.5e+03 |
| liner | liner+outflow | outflow.shift | 106 | -1.5e+03 | 28.6 | 500 | -1.5e+03 | 500 | 0.019 | 0.217 | 28.6 | -1.9e+03 | 1.5e+03 |
| passive | passive | host.ebv | 75 | 0 | 0.052 | 0.128 | 0 | 1.5 | 0.027 | 0 | 0.052 | -0.75 | 0.16 |
| passive | passive | host.sigma | 75 | 60.8 | 180 | 302 | 30 | 450 | 0 | 0 | 180 | 12 | 360 |
| passive | passive | host.velocity | 75 | -8.5 | 5.7 | 15.1 | -600 | 600 | 0 | 0 | 5.7 | -14 | 20 |
| passive | passive | narrow.fwhm | 75 | 50 | 600 | 600 | 50 | 600 | 0.12 | 0.507 | 600 | -230 | 880 |
| passive | passive | narrow.shift | 75 | -400 | -25.4 | 400 | -400 | 400 | 0.133 | 0.093 | -25.4 | -800 | 800 |
| sf | starforming | extinction.ebv | 104 | 0.004 | 0.31 | 0.762 | 0 | 2 | 0.01 | 0 | 0.31 | -0.15 | 0.92 |
| sf | starforming | host.ebv | 104 | 0.032 | 0.206 | 0.469 | 0 | 1.5 | 0.01 | 0 | 0.206 | -0.056 | 0.56 |
| sf | starforming | host.sigma | 104 | 33.9 | 75.8 | 286 | 30 | 450 | 0 | 0 | 75.8 | -17 | 340 |
| sf | starforming | host.velocity | 104 | -58.3 | -4.71 | 52.5 | -600 | 600 | 0 | 0 | -4.71 | -81 | 75 |
| sf | starforming | narrow.fwhm | 104 | 96.2 | 143 | 357 | 20 | 800 | 0 | 0 | 143 | 43 | 410 |
| sf | starforming | narrow.shift | 104 | -48.7 | -0.448 | 41 | -400 | 400 | 0 | 0 | -0.448 | -67 | 59 |
| sf | starforming | outflow.fwhm | 104 | 200 | 516 | 2.5e+03 | 200 | 2.5e+03 | 0.327 | 0.288 | 516 | -950 | 3.7e+03 |
| sf | starforming | outflow.shift | 104 | -800 | -75.6 | 400 | -800 | 400 | 0.087 | 0.115 | -75.6 | -1.4e+03 | 1e+03 |
| sf | starforming no outflow | extinction.ebv | 104 | 0.015 | 0.315 | 0.749 | 0 | 2 | 0.01 | 0 | 0.315 | -0.14 | 0.9 |
| sf | starforming no outflow | host.ebv | 104 | 0.022 | 0.2 | 0.468 | 0 | 1.5 | 0.01 | 0 | 0.2 | -0.067 | 0.56 |
| sf | starforming no outflow | host.sigma | 104 | 42 | 75.8 | 218 | 30 | 450 | 0 | 0 | 75.8 | 6.7 | 260 |
| sf | starforming no outflow | host.velocity | 104 | -54.9 | -6.15 | 55.3 | -600 | 600 | 0 | 0 | -6.15 | -77 | 78 |
| sf | starforming no outflow | narrow.fwhm | 104 | 98 | 147 | 429 | 20 | 800 | 0 | 0 | 147 | 31 | 500 |
| sf | starforming no outflow | narrow.shift | 104 | -13.4 | -0.431 | 9.85 | -400 | 400 | 0 | 0 | -0.431 | -19 | 15 |
| sy2 | seyfert2 | host.ebv | 104 | 0.027 | 0.171 | 0.426 | 0 | 1.5 | 0 | 0 | 0.171 | -0.053 | 0.51 |
| sy2 | seyfert2 | host.sigma | 104 | 46.8 | 128 | 289 | 30 | 450 | 0 | 0 | 128 | -1.7 | 340 |
| sy2 | seyfert2 | host.velocity | 104 | -79.2 | -2.94 | 41.9 | -600 | 600 | 0 | 0 | -2.94 | -110 | 67 |
| sy2 | seyfert2 | narrow.fwhm | 104 | 120 | 274 | 668 | 50 | 1.2e+03 | 0 | 0 | 274 | 10 | 780 |
| sy2 | seyfert2 | narrow.shift | 104 | -45 | 2.39 | 86.4 | -600 | 600 | 0 | 0 | 2.39 | -72 | 120 |
| sy2 | seyfert2 | narrow_hi.fwhm | 104 | 50 | 263 | 594 | 50 | 1.5e+03 | 0.029 | 0 | 263 | -680 | 710 |
| sy2 | seyfert2 | narrow_hi.shift | 104 | -173 | 6.03 | 216 | -600 | 600 | 0 | 0 | 6.03 | -260 | 300 |
| sy2 | seyfert2 | outflow.fwhm | 104 | 500 | 941 | 2.5e+03 | 500 | 2.5e+03 | 0.163 | 0.317 | 941 | -500 | 3.5e+03 |
| sy2 | seyfert2 | outflow.shift | 104 | -1.5e+03 | -74.6 | 500 | -1.5e+03 | 500 | 0.038 | 0.058 | -74.6 | -2.5e+03 | 1.5e+03 |
| sy2 | seyfert2 no outflow | host.ebv | 104 | 0.024 | 0.167 | 0.415 | 0 | 1.5 | 0 | 0 | 0.167 | -0.055 | 0.5 |
| sy2 | seyfert2 no outflow | host.sigma | 104 | 54 | 134 | 276 | 30 | 450 | 0 | 0 | 134 | 9.5 | 330 |
| sy2 | seyfert2 no outflow | host.velocity | 104 | -70.1 | -3.26 | 41.6 | -600 | 600 | 0 | 0 | -3.26 | -93 | 64 |
| sy2 | seyfert2 no outflow | narrow.fwhm | 104 | 155 | 335 | 952 | 50 | 1.2e+03 | 0 | 0.01 | 335 | -4.3 | 1.2e+03 |
| sy2 | seyfert2 no outflow | narrow.shift | 104 | -140 | 0.363 | 54.9 | -600 | 600 | 0 | 0 | 0.363 | -180 | 94 |
| sy2 | seyfert2 no outflow | narrow_hi.fwhm | 104 | 143 | 326 | 717 | 50 | 1.5e+03 | 0 | 0 | 326 | 28 | 840 |
| sy2 | seyfert2 no outflow | narrow_hi.shift | 104 | -77.4 | 4.35 | 146 | -600 | 600 | 0 | 0 | 4.35 | -130 | 200 |
| sy2 | seyfert2 one system | host.ebv | 104 | 0.028 | 0.172 | 0.424 | 0 | 1.5 | 0 | 0 | 0.172 | -0.052 | 0.51 |
| sy2 | seyfert2 one system | host.sigma | 104 | 57.4 | 131 | 288 | 30 | 450 | 0 | 0 | 131 | 11 | 340 |
| sy2 | seyfert2 one system | host.velocity | 104 | -81.1 | -1.92 | 41.5 | -600 | 600 | 0 | 0 | -1.92 | -110 | 67 |
| sy2 | seyfert2 one system | narrow.fwhm | 104 | 122 | 275 | 683 | 50 | 1.2e+03 | 0 | 0 | 275 | 9.5 | 800 |
| sy2 | seyfert2 one system | narrow.shift | 104 | -59.9 | 1.92 | 165 | -600 | 600 | 0 | 0 | 1.92 | -110 | 210 |
| sy2 | seyfert2 one system | outflow.fwhm | 104 | 500 | 852 | 2.5e+03 | 500 | 2.5e+03 | 0.279 | 0.317 | 852 | -500 | 3.5e+03 |
| sy2 | seyfert2 one system | outflow.shift | 104 | -1.5e+03 | -44.8 | 500 | -1.5e+03 | 500 | 0.019 | 0.058 | -44.8 | -1.9e+03 | 1.5e+03 |
| t1_civ | civ | broad.fwhm | 101 | 1.6e+03 | 6.21e+03 | 1.6e+04 | 1e+03 | 2e+04 | 0 | 0.01 | 6.21e+03 | -1.3e+03 | 1.9e+04 |
| t1_civ | civ | broad.shift | 101 | -3e+03 | -629 | 2.13e+03 | -3e+03 | 3e+03 | 0.04 | 0 | -629 | -6e+03 | 3.2e+03 |
| t1_civ | civ | continuum.index | 101 | -2.5 | -1.22 | 2.68 | -5 | 3 | 0 | 0 | -1.22 | -3.6 | 3.8 |
| t1_civ | civ | heii.fwhm | 101 | 1.44e+03 | 1.5e+04 | 1.5e+04 | 1e+03 | 1.5e+04 | 0 | 0.515 | 1.5e+04 | -1.3e+03 | 2.2e+04 |
| t1_civ | civ | heii.shift | 101 | -3e+03 | -3e+03 | 3e+03 | -3e+03 | 3e+03 | 0.693 | 0.02 | -3e+03 | -6e+03 | 4.2e+03 |
| t1_civ | civ 2 broad | broad.fwhm | 101 | 1e+03 | 3.02e+03 | 6.03e+03 | 1e+03 | 7.12e+03 | 0.05 | 0 | 3.02e+03 | -2.1e+03 | 7.1e+03 |
| t1_civ | civ 2 broad | broad.shift | 101 | -1.96e+03 | -374 | 2.87e+03 | -3e+03 | 3e+03 | 0 | 0.01 | -374 | -3e+03 | 3.9e+03 |
| t1_civ | civ 2 broad | broad2.fwhm | 101 | 4.21e+03 | 1.05e+04 | 3e+04 | 4.21e+03 | 3e+04 | 0.079 | 0.03 | 1.05e+04 | -8.7e+03 | 4.3e+04 |
| t1_civ | civ 2 broad | broad2.shift | 101 | -5e+03 | -790 | 4.71e+03 | -5e+03 | 5e+03 | 0.079 | 0.01 | -790 | -1e+04 | 6.7e+03 |
| t1_civ | civ 2 broad | continuum.index | 101 | -3.81 | -0.967 | 2.89 | -5 | 3 | 0.01 | 0.01 | -0.967 | -5.2 | 4.3 |
| t1_civ | civ 2 broad | heii.fwhm | 101 | 1.67e+03 | 1.14e+04 | 1.5e+04 | 1e+03 | 1.5e+04 | 0 | 0.218 | 1.14e+04 | -1e+03 | 2.2e+04 |
| t1_civ | civ 2 broad | heii.shift | 101 | -3e+03 | -2.88e+03 | 3e+03 | -3e+03 | 3e+03 | 0.495 | 0.03 | -2.88e+03 | -6e+03 | 6e+03 |
| t1_hb | hbeta | broad.fwhm | 139 | 1e+03 | 3.24e+03 | 6.97e+03 | 1e+03 | 7.12e+03 | 0.014 | 0.007 | 3.24e+03 | -200 | 8.2e+03 |
| t1_hb | hbeta | broad.shift | 139 | -3e+03 | 11.7 | 1.38e+03 | -3e+03 | 3e+03 | 0.022 | 0 | 11.7 | -6e+03 | 2.3e+03 |
| t1_hb | hbeta | broad2.fwhm | 139 | 4.21e+03 | 1.37e+04 | 3e+04 | 4.21e+03 | 3e+04 | 0.022 | 0.022 | 1.37e+04 | -8.7e+03 | 4.3e+04 |
| t1_hb | hbeta | broad2.shift | 139 | -2.51e+03 | 1.81e+03 | 5e+03 | -5e+03 | 5e+03 | 0 | 0.18 | 1.81e+03 | -4.1e+03 | 1e+04 |
| t1_hb | hbeta | continuum.index | 139 | -4.76 | -0.976 | 3 | -5 | 3 | 0.007 | 0.122 | -0.976 | -6.4 | 7 |
| t1_hb | hbeta | feii.fwhm | 139 | 1e+03 | 3.16e+03 | 8e+03 | 1e+03 | 8e+03 | 0.029 | 0.05 | 3.16e+03 | -2.5e+03 | 1.2e+04 |
| t1_hb | hbeta | feii.shift | 139 | -1.99e+03 | -181 | 2e+03 | -2e+03 | 2e+03 | 0.007 | 0.058 | -181 | -2.8e+03 | 4e+03 |
| t1_hb | hbeta | heii.fwhm | 139 | 1e+03 | 5.4e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.036 | 0.058 | 5.4e+03 | -6e+03 | 2.2e+04 |
| t1_hb | hbeta | heii.shift | 139 | -4e+03 | -160 | 2e+03 | -4e+03 | 2e+03 | 0.058 | 0.151 | -160 | -7e+03 | 5e+03 |
| t1_hb | hbeta | narrow.fwhm | 139 | 50 | 531 | 1.2e+03 | 50 | 1.2e+03 | 0.014 | 0.072 | 531 | -180 | 1.8e+03 |
| t1_hb | hbeta | narrow.shift | 139 | -487 | -13.3 | 424 | -600 | 600 | 0 | 0.007 | -13.3 | -670 | 610 |
| t1_hb | hbeta 1 broad | broad.fwhm | 139 | 2.11e+03 | 5.19e+03 | 2e+04 | 1e+03 | 2e+04 | 0 | 0.036 | 5.19e+03 | -1.5e+03 | 3e+04 |
| t1_hb | hbeta 1 broad | broad.shift | 139 | -2.4e+03 | 192 | 3e+03 | -3e+03 | 3e+03 | 0 | 0.014 | 192 | -3.5e+03 | 4.1e+03 |
| t1_hb | hbeta 1 broad | continuum.index | 139 | -2.78 | -1.13 | 3 | -5 | 3 | 0 | 0.072 | -1.13 | -4 | 7 |
| t1_hb | hbeta 1 broad | feii.fwhm | 139 | 1.13e+03 | 4.63e+03 | 8e+03 | 1e+03 | 8e+03 | 0.007 | 0.165 | 4.63e+03 | -250 | 1.2e+04 |
| t1_hb | hbeta 1 broad | feii.shift | 139 | -2e+03 | -15.7 | 2e+03 | -2e+03 | 2e+03 | 0.072 | 0.065 | -15.7 | -4e+03 | 4e+03 |
| t1_hb | hbeta 1 broad | heii.fwhm | 139 | 1e+03 | 7.47e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.079 | 0.273 | 7.47e+03 | -6e+03 | 2.2e+04 |
| t1_hb | hbeta 1 broad | heii.shift | 139 | -4e+03 | 375 | 2e+03 | -4e+03 | 2e+03 | 0.072 | 0.374 | 375 | -7e+03 | 5e+03 |
| t1_hb | hbeta 1 broad | narrow.fwhm | 139 | 199 | 590 | 1.2e+03 | 50 | 1.2e+03 | 0 | 0.18 | 590 | -0.64 | 1.8e+03 |
| t1_hb | hbeta 1 broad | narrow.shift | 139 | -517 | -16.6 | 600 | -600 | 600 | 0 | 0.014 | -16.6 | -750 | 830 |
| t1_hb | hbeta 3 broad | broad.fwhm | 139 | 1e+03 | 2.09e+03 | 4.04e+03 | 1e+03 | 4.04e+03 | 0.115 | 0.086 | 2.09e+03 | -520 | 5.6e+03 |
| t1_hb | hbeta 3 broad | broad.shift | 139 | -3e+03 | 129 | 3e+03 | -3e+03 | 3e+03 | 0.022 | 0.022 | 129 | -6e+03 | 6e+03 |
| t1_hb | hbeta 3 broad | broad2.fwhm | 139 | 2.39e+03 | 4.27e+03 | 1.26e+04 | 2.39e+03 | 1.26e+04 | 0.158 | 0.065 | 4.27e+03 | -2.7e+03 | 1.8e+04 |
| t1_hb | hbeta 3 broad | broad2.shift | 139 | -4.86e+03 | -16.9 | 5e+03 | -5e+03 | 5e+03 | 0.007 | 0.115 | -16.9 | -6.9e+03 | 1e+04 |
| t1_hb | hbeta 3 broad | broad3.fwhm | 139 | 7.43e+03 | 1.34e+04 | 3e+04 | 7.43e+03 | 3e+04 | 0.108 | 0.022 | 1.34e+04 | -3.9e+03 | 4.2e+04 |
| t1_hb | hbeta 3 broad | broad3.shift | 139 | -5e+03 | 2.16e+03 | 5e+03 | -5e+03 | 5e+03 | 0.036 | 0.223 | 2.16e+03 | -1e+04 | 1e+04 |
| t1_hb | hbeta 3 broad | continuum.index | 139 | -3.71 | -0.947 | 3 | -5 | 3 | 0 | 0.122 | -0.947 | -5.1 | 7 |
| t1_hb | hbeta 3 broad | feii.fwhm | 139 | 1e+03 | 3.06e+03 | 8e+03 | 1e+03 | 8e+03 | 0.043 | 0.029 | 3.06e+03 | -2.5e+03 | 1.2e+04 |
| t1_hb | hbeta 3 broad | feii.shift | 139 | -1.87e+03 | -230 | 2e+03 | -2e+03 | 2e+03 | 0.007 | 0.029 | -230 | -2.7e+03 | 4e+03 |
| t1_hb | hbeta 3 broad | heii.fwhm | 139 | 1e+03 | 5.84e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.065 | 0.072 | 5.84e+03 | -6e+03 | 2.2e+04 |
| t1_hb | hbeta 3 broad | heii.shift | 139 | -4e+03 | -363 | 2e+03 | -4e+03 | 2e+03 | 0.065 | 0.086 | -363 | -7e+03 | 5e+03 |
| t1_hb | hbeta 3 broad | narrow.fwhm | 139 | 50 | 517 | 1.2e+03 | 50 | 1.2e+03 | 0.014 | 0.072 | 517 | -180 | 1.8e+03 |
| t1_hb | hbeta 3 broad | narrow.shift | 139 | -482 | -12.9 | 576 | -600 | 600 | 0 | 0.007 | -12.9 | -700 | 790 |
| t1_hb | hbeta+outflow | broad.fwhm | 139 | 1.13e+03 | 3.26e+03 | 7.12e+03 | 1e+03 | 7.12e+03 | 0.007 | 0.029 | 3.26e+03 | -64 | 1.1e+04 |
| t1_hb | hbeta+outflow | broad.shift | 139 | -3e+03 | 63.1 | 1.69e+03 | -3e+03 | 3e+03 | 0.014 | 0 | 63.1 | -4e+03 | 2.7e+03 |
| t1_hb | hbeta+outflow | broad2.fwhm | 139 | 4.21e+03 | 1.16e+04 | 3e+04 | 4.21e+03 | 3e+04 | 0.065 | 0.043 | 1.16e+04 | -8.7e+03 | 4.3e+04 |
| t1_hb | hbeta+outflow | broad2.shift | 139 | -4.5e+03 | 840 | 5e+03 | -5e+03 | 5e+03 | 0 | 0.086 | 840 | -6.4e+03 | 1e+04 |
| t1_hb | hbeta+outflow | continuum.index | 139 | -4.03 | -1.01 | 3 | -5 | 3 | 0.007 | 0.058 | -1.01 | -5.5 | 7 |
| t1_hb | hbeta+outflow | feii.fwhm | 139 | 1.04e+03 | 3.55e+03 | 8e+03 | 1e+03 | 8e+03 | 0.007 | 0.072 | 3.55e+03 | -360 | 1.2e+04 |
| t1_hb | hbeta+outflow | feii.shift | 139 | -2e+03 | 102 | 2e+03 | -2e+03 | 2e+03 | 0.022 | 0.079 | 102 | -4e+03 | 4e+03 |
| t1_hb | hbeta+outflow | heii.fwhm | 139 | 1e+03 | 4.72e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.101 | 0.072 | 4.72e+03 | -6e+03 | 2.2e+04 |
| t1_hb | hbeta+outflow | heii.shift | 139 | -4e+03 | -20.3 | 2e+03 | -4e+03 | 2e+03 | 0.036 | 0.165 | -20.3 | -7e+03 | 5e+03 |
| t1_hb | hbeta+outflow | narrow.fwhm | 139 | 50 | 378 | 1.2e+03 | 50 | 1.2e+03 | 0.022 | 0.108 | 378 | -530 | 1.8e+03 |
| t1_hb | hbeta+outflow | narrow.shift | 139 | -511 | 19.1 | 554 | -600 | 600 | 0.007 | 0.007 | 19.1 | -730 | 770 |
| t1_hb | hbeta+outflow | outflow.fwhm | 139 | 300 | 794 | 2.5e+03 | 300 | 2.5e+03 | 0.079 | 0.065 | 794 | -800 | 3.6e+03 |
| t1_hb | hbeta+outflow | outflow.shift | 139 | -2e+03 | -180 | 200 | -2e+03 | 200 | 0.014 | 0.058 | -180 | -2.5e+03 | 1.3e+03 |
| t1_low | halpha | broad.fwhm | 211 | 1e+03 | 2.41e+03 | 7.11e+03 | 1e+03 | 7.12e+03 | 0.028 | 0.009 | 2.41e+03 | -2.1e+03 | 8.4e+03 |
| t1_low | halpha | broad.shift | 211 | -2.68e+03 | 71.4 | 1.45e+03 | -3e+03 | 3e+03 | 0.005 | 0 | 71.4 | -3.5e+03 | 2.3e+03 |
| t1_low | halpha | broad2.fwhm | 211 | 4.21e+03 | 1.02e+04 | 3e+04 | 4.21e+03 | 3e+04 | 0.076 | 0.033 | 1.02e+04 | -8.7e+03 | 4.3e+04 |
| t1_low | halpha | broad2.shift | 211 | -5e+03 | -436 | 5e+03 | -5e+03 | 5e+03 | 0.194 | 0.014 | -436 | -1e+04 | 7e+03 |
| t1_low | halpha | continuum.index | 211 | -2.26 | -0.26 | 1.38 | -5 | 3 | 0 | 0 | -0.26 | -3 | 2.2 |
| t1_low | halpha | narrow.fwhm | 211 | 126 | 344 | 1.2e+03 | 50 | 1.2e+03 | 0 | 0.047 | 344 | -89 | 1.8e+03 |
| t1_low | halpha | narrow.shift | 211 | -587 | 20.7 | 288 | -600 | 600 | 0.009 | 0 | 20.7 | -770 | 470 |
| t1_low | halpha 1 broad | broad.fwhm | 211 | 1.5e+03 | 3.75e+03 | 1.93e+04 | 1e+03 | 2e+04 | 0.005 | 0.005 | 3.75e+03 | -2.1e+03 | 2.3e+04 |
| t1_low | halpha 1 broad | broad.shift | 211 | -3e+03 | 50.2 | 852 | -3e+03 | 3e+03 | 0.033 | 0 | 50.2 | -6e+03 | 1.7e+03 |
| t1_low | halpha 1 broad | continuum.index | 211 | -2.07 | -0.376 | 1 | -5 | 3 | 0 | 0 | -0.376 | -2.7 | 1.7 |
| t1_low | halpha 1 broad | narrow.fwhm | 211 | 125 | 391 | 1.2e+03 | 50 | 1.2e+03 | 0 | 0.095 | 391 | -91 | 1.8e+03 |
| t1_low | halpha 1 broad | narrow.shift | 211 | -598 | 9.3 | 283 | -600 | 600 | 0.009 | 0.005 | 9.3 | -780 | 460 |
| t1_low | halpha+outflow | broad.fwhm | 211 | 1e+03 | 3.33e+03 | 7.01e+03 | 1e+03 | 7.12e+03 | 0.038 | 0.005 | 3.33e+03 | -2.1e+03 | 8.3e+03 |
| t1_low | halpha+outflow | broad.shift | 211 | -2.99e+03 | 170 | 2.53e+03 | -3e+03 | 3e+03 | 0.009 | 0.005 | 170 | -4.1e+03 | 3.7e+03 |
| t1_low | halpha+outflow | broad2.fwhm | 211 | 4.21e+03 | 1.31e+04 | 3e+04 | 4.21e+03 | 3e+04 | 0.024 | 0.033 | 1.31e+04 | -8.7e+03 | 4.3e+04 |
| t1_low | halpha+outflow | broad2.shift | 211 | -5e+03 | -3.08e+03 | 4.35e+03 | -5e+03 | 5e+03 | 0.336 | 0.005 | -3.08e+03 | -1e+04 | 6.3e+03 |
| t1_low | halpha+outflow | continuum.index | 211 | -1.97 | -0.106 | 1.39 | -5 | 3 | 0 | 0 | -0.106 | -2.7 | 2.1 |
| t1_low | halpha+outflow | narrow.fwhm | 211 | 89.7 | 315 | 1.2e+03 | 50 | 1.2e+03 | 0.005 | 0.024 | 315 | -140 | 1.8e+03 |
| t1_low | halpha+outflow | narrow.shift | 211 | -437 | 31.2 | 465 | -600 | 600 | 0.005 | 0 | 31.2 | -620 | 650 |
| t1_low | halpha+outflow | outflow.fwhm | 211 | 300 | 1.02e+03 | 2e+03 | 300 | 2e+03 | 0.152 | 0.1 | 1.02e+03 | -550 | 2.9e+03 |
| t1_low | halpha+outflow | outflow.shift | 211 | -1.99e+03 | -150 | 200 | -2e+03 | 200 | 0.009 | 0.052 | -150 | -2.5e+03 | 1.3e+03 |
| t1_low | hbeta | broad.fwhm | 211 | 1e+03 | 2.94e+03 | 7.09e+03 | 1e+03 | 7.12e+03 | 0.038 | 0.005 | 2.94e+03 | -2.1e+03 | 8.4e+03 |
| t1_low | hbeta | broad.shift | 211 | -3e+03 | 50 | 1.62e+03 | -3e+03 | 3e+03 | 0.052 | 0 | 50 | -6e+03 | 2.6e+03 |
| t1_low | hbeta | broad2.fwhm | 211 | 4.21e+03 | 1.66e+04 | 3e+04 | 4.21e+03 | 3e+04 | 0.014 | 0.095 | 1.66e+04 | -950 | 4.3e+04 |
| t1_low | hbeta | broad2.shift | 211 | -4.33e+03 | 1.88e+03 | 5e+03 | -5e+03 | 5e+03 | 0.005 | 0.218 | 1.88e+03 | -6.2e+03 | 1e+04 |
| t1_low | hbeta | continuum.index | 211 | -1.93 | 1.34 | 3 | -5 | 3 | 0 | 0.408 | 1.34 | -3 | 7 |
| t1_low | hbeta | feii.fwhm | 211 | 1e+03 | 5.53e+03 | 8e+03 | 1e+03 | 8e+03 | 0.019 | 0.114 | 5.53e+03 | -400 | 1.2e+04 |
| t1_low | hbeta | feii.shift | 211 | -1.99e+03 | 262 | 2e+03 | -2e+03 | 2e+03 | 0.009 | 0.237 | 262 | -2.8e+03 | 4e+03 |
| t1_low | hbeta | heii.fwhm | 211 | 1e+03 | 5.05e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.066 | 0.109 | 5.05e+03 | -6e+03 | 2.2e+04 |
| t1_low | hbeta | heii.shift | 211 | -4e+03 | 155 | 2e+03 | -4e+03 | 2e+03 | 0.062 | 0.251 | 155 | -7e+03 | 5e+03 |
| t1_low | hbeta | narrow.fwhm | 211 | 181 | 417 | 938 | 50 | 1.2e+03 | 0 | 0.005 | 417 | 29 | 1.1e+03 |
| t1_low | hbeta | narrow.shift | 211 | -207 | -7.76 | 201 | -600 | 600 | 0 | 0 | -7.76 | -290 | 290 |
| t1_low | hbeta 1 broad | broad.fwhm | 211 | 1.02e+03 | 7.19e+03 | 2e+04 | 1e+03 | 2e+04 | 0.009 | 0.265 | 7.19e+03 | -2.8e+03 | 3e+04 |
| t1_low | hbeta 1 broad | broad.shift | 211 | -2.36e+03 | 184 | 3e+03 | -3e+03 | 3e+03 | 0.005 | 0.171 | 184 | -3.5e+03 | 6e+03 |
| t1_low | hbeta 1 broad | continuum.index | 211 | -2 | 0.127 | 3 | -5 | 3 | 0 | 0.152 | 0.127 | -3.1 | 7 |
| t1_low | hbeta 1 broad | feii.fwhm | 211 | 1e+03 | 4.94e+03 | 8e+03 | 1e+03 | 8e+03 | 0.019 | 0.156 | 4.94e+03 | -400 | 1.2e+04 |
| t1_low | hbeta 1 broad | feii.shift | 211 | -2e+03 | 376 | 2e+03 | -2e+03 | 2e+03 | 0.081 | 0.332 | 376 | -4e+03 | 4e+03 |
| t1_low | hbeta 1 broad | heii.fwhm | 211 | 1e+03 | 7.12e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.09 | 0.213 | 7.12e+03 | -6e+03 | 2.2e+04 |
| t1_low | hbeta 1 broad | heii.shift | 211 | -4e+03 | 919 | 2e+03 | -4e+03 | 2e+03 | 0.014 | 0.327 | 919 | -5.2e+03 | 5e+03 |
| t1_low | hbeta 1 broad | narrow.fwhm | 211 | 185 | 444 | 1.2e+03 | 50 | 1.2e+03 | 0 | 0.019 | 444 | -18 | 1.5e+03 |
| t1_low | hbeta 1 broad | narrow.shift | 211 | -216 | -11.5 | 142 | -600 | 600 | 0 | 0 | -11.5 | -290 | 220 |
| t1_low | hbeta 3 broad | broad.fwhm | 211 | 1e+03 | 2e+03 | 4.04e+03 | 1e+03 | 4.04e+03 | 0.128 | 0.028 | 2e+03 | -520 | 5.6e+03 |
| t1_low | hbeta 3 broad | broad.shift | 211 | -3e+03 | 28.6 | 2.9e+03 | -3e+03 | 3e+03 | 0.019 | 0 | 28.6 | -4.2e+03 | 4.1e+03 |
| t1_low | hbeta 3 broad | broad2.fwhm | 211 | 2.39e+03 | 3.47e+03 | 1.26e+04 | 2.39e+03 | 1.26e+04 | 0.213 | 0.043 | 3.47e+03 | -2.7e+03 | 1.8e+04 |
| t1_low | hbeta 3 broad | broad2.shift | 211 | -5e+03 | 295 | 5e+03 | -5e+03 | 5e+03 | 0.066 | 0.118 | 295 | -1e+04 | 1e+04 |
| t1_low | hbeta 3 broad | broad3.fwhm | 211 | 7.43e+03 | 1.73e+04 | 3e+04 | 7.43e+03 | 3e+04 | 0.076 | 0.104 | 1.73e+04 | -3.9e+03 | 4.2e+04 |
| t1_low | hbeta 3 broad | broad3.shift | 211 | -5e+03 | 1.53e+03 | 5e+03 | -5e+03 | 5e+03 | 0.057 | 0.265 | 1.53e+03 | -1e+04 | 1e+04 |
| t1_low | hbeta 3 broad | continuum.index | 211 | -1.95 | 0.982 | 3 | -5 | 3 | 0 | 0.403 | 0.982 | -3 | 7 |
| t1_low | hbeta 3 broad | feii.fwhm | 211 | 1e+03 | 5.28e+03 | 8e+03 | 1e+03 | 8e+03 | 0.019 | 0.175 | 5.28e+03 | -400 | 1.2e+04 |
| t1_low | hbeta 3 broad | feii.shift | 211 | -1.99e+03 | 558 | 2e+03 | -2e+03 | 2e+03 | 0.009 | 0.332 | 558 | -2.8e+03 | 4e+03 |
| t1_low | hbeta 3 broad | heii.fwhm | 211 | 1e+03 | 4.73e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.09 | 0.052 | 4.73e+03 | -6e+03 | 2.2e+04 |
| t1_low | hbeta 3 broad | heii.shift | 211 | -4e+03 | -32.8 | 2e+03 | -4e+03 | 2e+03 | 0.1 | 0.137 | -32.8 | -7e+03 | 5e+03 |
| t1_low | hbeta 3 broad | narrow.fwhm | 211 | 161 | 412 | 941 | 50 | 1.2e+03 | 0.005 | 0.005 | 412 | 4.5 | 1.1e+03 |
| t1_low | hbeta 3 broad | narrow.shift | 211 | -208 | -6.95 | 200 | -600 | 600 | 0 | 0 | -6.95 | -290 | 290 |
| t1_low | hbeta+outflow | broad.fwhm | 211 | 1e+03 | 2.77e+03 | 7.1e+03 | 1e+03 | 7.12e+03 | 0.076 | 0.009 | 2.77e+03 | -2.1e+03 | 8.4e+03 |
| t1_low | hbeta+outflow | broad.shift | 211 | -3e+03 | 83.3 | 2.65e+03 | -3e+03 | 3e+03 | 0.019 | 0 | 83.3 | -4.2e+03 | 3.8e+03 |
| t1_low | hbeta+outflow | broad2.fwhm | 211 | 4.21e+03 | 1.68e+04 | 3e+04 | 4.21e+03 | 3e+04 | 0.014 | 0.109 | 1.68e+04 | -950 | 4.3e+04 |
| t1_low | hbeta+outflow | broad2.shift | 211 | -5e+03 | 720 | 5e+03 | -5e+03 | 5e+03 | 0.024 | 0.128 | 720 | -1e+04 | 1e+04 |
| t1_low | hbeta+outflow | continuum.index | 211 | -2.08 | 0.685 | 3 | -5 | 3 | 0 | 0.37 | 0.685 | -3.1 | 7 |
| t1_low | hbeta+outflow | feii.fwhm | 211 | 1.36e+03 | 5.14e+03 | 8e+03 | 1e+03 | 8e+03 | 0 | 0.09 | 5.14e+03 | 26 | 1.2e+04 |
| t1_low | hbeta+outflow | feii.shift | 211 | -1.96e+03 | 509 | 2e+03 | -2e+03 | 2e+03 | 0.005 | 0.308 | 509 | -2.8e+03 | 4e+03 |
| t1_low | hbeta+outflow | heii.fwhm | 211 | 1e+03 | 4.54e+03 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.09 | 0.076 | 4.54e+03 | -6e+03 | 2.2e+04 |
| t1_low | hbeta+outflow | heii.shift | 211 | -4e+03 | 103 | 2e+03 | -4e+03 | 2e+03 | 0.118 | 0.185 | 103 | -7e+03 | 5e+03 |
| t1_low | hbeta+outflow | narrow.fwhm | 211 | 50 | 311 | 1.2e+03 | 50 | 1.2e+03 | 0.014 | 0.028 | 311 | -180 | 1.8e+03 |
| t1_low | hbeta+outflow | narrow.shift | 211 | -587 | 15.7 | 227 | -600 | 600 | 0.009 | 0 | 15.7 | -760 | 400 |
| t1_low | hbeta+outflow | outflow.fwhm | 211 | 300 | 650 | 2.5e+03 | 300 | 2.5e+03 | 0.156 | 0.024 | 650 | -800 | 3.6e+03 |
| t1_low | hbeta+outflow | outflow.shift | 211 | -1.99e+03 | -170 | 200 | -2e+03 | 200 | 0.009 | 0.024 | -170 | -2.5e+03 | 1.3e+03 |
| t1_low | optical | broad.fwhm | 211 | 1e+03 | 3.14e+03 | 7.12e+03 | 1e+03 | 7.12e+03 | 0.005 | 0.171 | 3.14e+03 | -220 | 1.1e+04 |
| t1_low | optical | broad.shift | 211 | -3e+03 | 67.3 | 3e+03 | -3e+03 | 3e+03 | 0.137 | 0.033 | 67.3 | -6e+03 | 6e+03 |
| t1_low | optical | broad2.fwhm | 211 | 4.21e+03 | 1.31e+04 | 2.53e+04 | 4.21e+03 | 3e+04 | 0.043 | 0 | 1.31e+04 | -8.7e+03 | 3e+04 |
| t1_low | optical | broad2.shift | 211 | -5e+03 | 2.26e+03 | 5e+03 | -5e+03 | 5e+03 | 0.043 | 0.28 | 2.26e+03 | -1e+04 | 1e+04 |
| t1_low | optical | broad_ha.fwhm | 211 | 1e+03 | 2.11e+03 | 7.1e+03 | 1e+03 | 7.12e+03 | 0.09 | 0.009 | 2.11e+03 | -2.1e+03 | 8.4e+03 |
| t1_low | optical | broad_ha.shift | 211 | -1.77e+03 | 43.1 | 2.99e+03 | -3e+03 | 3e+03 | 0 | 0.009 | 43.1 | -2.8e+03 | 4e+03 |
| t1_low | optical | broad_ha2.fwhm | 211 | 4.21e+03 | 5.46e+03 | 3e+04 | 4.21e+03 | 3e+04 | 0.303 | 0.047 | 5.46e+03 | -8.7e+03 | 4.3e+04 |
| t1_low | optical | broad_ha2.shift | 211 | -5e+03 | 122 | 5e+03 | -5e+03 | 5e+03 | 0.076 | 0.019 | 122 | -1e+04 | 7e+03 |
| t1_low | optical | broad_he.fwhm | 211 | 1.62e+03 | 1.5e+04 | 1.5e+04 | 1e+03 | 1.5e+04 | 0.005 | 0.73 | 1.5e+04 | -1.1e+03 | 2.2e+04 |
| t1_low | optical | broad_he.shift | 211 | -4e+03 | -720 | 2e+03 | -4e+03 | 2e+03 | 0.095 | 0.043 | -720 | -7e+03 | 5e+03 |
| t1_low | optical | continuum.index_blue | 211 | -3.81 | 0.143 | 3 | -5 | 3 | 0 | 0.284 | 0.143 | -5.2 | 7 |
| t1_low | optical | continuum.index_red | 211 | -1.72 | -0.054 | 1.69 | -5 | 3 | 0 | 0 | -0.054 | -2.5 | 2.4 |
| t1_low | optical | feii.fwhm | 211 | 1e+03 | 5.85e+03 | 8e+03 | 1e+03 | 8e+03 | 0.047 | 0.313 | 5.85e+03 | -2.5e+03 | 1.2e+04 |
| t1_low | optical | feii.shift | 211 | -2e+03 | -128 | 2e+03 | -2e+03 | 2e+03 | 0.275 | 0.19 | -128 | -4e+03 | 4e+03 |
| t1_low | optical | narrow.fwhm | 211 | 146 | 398 | 1.05e+03 | 50 | 1.2e+03 | 0 | 0.005 | 398 | -36 | 1.3e+03 |
| t1_low | optical | narrow.shift | 211 | -203 | 4.91 | 218 | -600 | 600 | 0 | 0 | 4.91 | -290 | 310 |
| t1_low | optical+outflow | broad.fwhm | 211 | 1e+03 | 3.11e+03 | 7.12e+03 | 1e+03 | 7.12e+03 | 0.028 | 0.166 | 3.11e+03 | -2.1e+03 | 1.1e+04 |
| t1_low | optical+outflow | broad.shift | 211 | -3e+03 | 91.3 | 3e+03 | -3e+03 | 3e+03 | 0.109 | 0.038 | 91.3 | -6e+03 | 6e+03 |
| t1_low | optical+outflow | broad2.fwhm | 211 | 4.21e+03 | 1.29e+04 | 2.85e+04 | 4.21e+03 | 3e+04 | 0.071 | 0 | 1.29e+04 | -8.7e+03 | 3.4e+04 |
| t1_low | optical+outflow | broad2.shift | 211 | -5e+03 | 838 | 5e+03 | -5e+03 | 5e+03 | 0.076 | 0.142 | 838 | -1e+04 | 1e+04 |
| t1_low | optical+outflow | broad_ha.fwhm | 211 | 1e+03 | 2.12e+03 | 7.12e+03 | 1e+03 | 7.12e+03 | 0.057 | 0.014 | 2.12e+03 | -2.1e+03 | 8.4e+03 |
| t1_low | optical+outflow | broad_ha.shift | 211 | -2.73e+03 | 63.8 | 2.99e+03 | -3e+03 | 3e+03 | 0 | 0.009 | 63.8 | -3.9e+03 | 4.2e+03 |
| t1_low | optical+outflow | broad_ha2.fwhm | 211 | 4.21e+03 | 5.47e+03 | 3e+04 | 4.21e+03 | 3e+04 | 0.261 | 0.047 | 5.47e+03 | -8.7e+03 | 4.3e+04 |
| t1_low | optical+outflow | broad_ha2.shift | 211 | -5e+03 | 150 | 5e+03 | -5e+03 | 5e+03 | 0.085 | 0.047 | 150 | -1e+04 | 1e+04 |
| t1_low | optical+outflow | broad_he.fwhm | 211 | 1.63e+03 | 1.5e+04 | 1.5e+04 | 1e+03 | 1.5e+04 | 0 | 0.763 | 1.5e+04 | -1.1e+03 | 2.2e+04 |
| t1_low | optical+outflow | broad_he.shift | 211 | -4e+03 | -769 | 2e+03 | -4e+03 | 2e+03 | 0.081 | 0.047 | -769 | -7e+03 | 5e+03 |
| t1_low | optical+outflow | continuum.index_blue | 211 | -3.88 | 0.235 | 3 | -5 | 3 | 0 | 0.294 | 0.235 | -5.3 | 7 |
| t1_low | optical+outflow | continuum.index_red | 211 | -1.69 | -0.049 | 1.68 | -5 | 3 | 0 | 0 | -0.049 | -2.4 | 2.4 |
| t1_low | optical+outflow | feii.fwhm | 211 | 1.41e+03 | 6.27e+03 | 8e+03 | 1e+03 | 8e+03 | 0 | 0.346 | 6.27e+03 | 95 | 1.2e+04 |
| t1_low | optical+outflow | feii.shift | 211 | -2e+03 | 97 | 2e+03 | -2e+03 | 2e+03 | 0.294 | 0.223 | 97 | -4e+03 | 4e+03 |
| t1_low | optical+outflow | narrow.fwhm | 211 | 125 | 323 | 1.2e+03 | 50 | 1.2e+03 | 0 | 0.038 | 323 | -91 | 1.8e+03 |
| t1_low | optical+outflow | narrow.shift | 211 | -407 | 20.2 | 237 | -600 | 600 | 0.005 | 0 | 20.2 | -540 | 370 |
| t1_low | optical+outflow | outflow.fwhm | 211 | 300 | 629 | 2.5e+03 | 300 | 2.5e+03 | 0.123 | 0.019 | 629 | -800 | 3e+03 |
| t1_low | optical+outflow | outflow.shift | 211 | -1.98e+03 | -109 | 200 | -2e+03 | 200 | 0.009 | 0.028 | -109 | -2.5e+03 | 1.3e+03 |
| t1_mgii | mgii | broad.fwhm | 135 | 1.16e+03 | 4.12e+03 | 8.67e+03 | 1e+03 | 2e+04 | 0 | 0 | 4.12e+03 | -350 | 1.1e+04 |
| t1_mgii | mgii | broad.shift | 135 | -1.34e+03 | 66.3 | 1.18e+03 | -3e+03 | 3e+03 | 0 | 0 | 66.3 | -1.9e+03 | 1.7e+03 |
| t1_mgii | mgii | continuum.index | 135 | -5 | -2 | 0.668 | -5 | 3 | 0.385 | 0 | -2 | -9 | 1.9 |
| t1_mgii | mgii | feii_uv.fwhm | 135 | 2.31e+03 | 3.97e+03 | 9.57e+03 | 1e+03 | 1e+04 | 0 | 0.007 | 3.97e+03 | 860 | 1.2e+04 |
| t1_mgii | mgii | feii_uv.shift | 135 | -2e+03 | 333 | 2e+03 | -2e+03 | 2e+03 | 0.03 | 0.015 | 333 | -4e+03 | 2.8e+03 |
| t1_mgii | mgii | narrow.fwhm | 135 | 50 | 1.2e+03 | 1.2e+03 | 50 | 1.2e+03 | 0.104 | 0.526 | 1.2e+03 | -530 | 1.8e+03 |
| t1_mgii | mgii | narrow.shift | 135 | -600 | -8.3 | 600 | -600 | 600 | 0.067 | 0.126 | -8.3 | -1.2e+03 | 1.2e+03 |
| t1_mgii | mgii 2 broad | broad.fwhm | 135 | 1e+03 | 2.66e+03 | 5.79e+03 | 1e+03 | 7.12e+03 | 0.074 | 0 | 2.66e+03 | -2.1e+03 | 6.8e+03 |
| t1_mgii | mgii 2 broad | broad.shift | 135 | -2.33e+03 | 97.9 | 3e+03 | -3e+03 | 3e+03 | 0 | 0.015 | 97.9 | -3.4e+03 | 4.1e+03 |
| t1_mgii | mgii 2 broad | broad2.fwhm | 135 | 4.21e+03 | 8.2e+03 | 3e+04 | 4.21e+03 | 3e+04 | 0.156 | 0.03 | 8.2e+03 | -8.7e+03 | 4.3e+04 |
| t1_mgii | mgii 2 broad | broad2.shift | 135 | -5e+03 | -168 | 5e+03 | -5e+03 | 5e+03 | 0.037 | 0.052 | -168 | -1e+04 | 1e+04 |
| t1_mgii | mgii 2 broad | continuum.index | 135 | -5 | -5 | 0.782 | -5 | 3 | 0.511 | 0 | -5 | -9 | 2 |
| t1_mgii | mgii 2 broad | feii_uv.fwhm | 135 | 1.23e+03 | 3.49e+03 | 1e+04 | 1e+03 | 1e+04 | 0.007 | 0.015 | 3.49e+03 | -530 | 1.2e+04 |
| t1_mgii | mgii 2 broad | feii_uv.shift | 135 | -2e+03 | 107 | 1.88e+03 | -2e+03 | 2e+03 | 0.067 | 0 | 107 | -4e+03 | 2.7e+03 |
| t1_mgii | mgii 2 broad | narrow.fwhm | 135 | 50 | 544 | 1.2e+03 | 50 | 1.2e+03 | 0.17 | 0.296 | 544 | -530 | 1.8e+03 |
| t1_mgii | mgii 2 broad | narrow.shift | 135 | -600 | -59.1 | 600 | -600 | 600 | 0.096 | 0.141 | -59.1 | -1.2e+03 | 1.2e+03 |

## Classification against the SDSS/MPA-JHU groups (agreement 63%)

| group | composite | liner | passive | retired | seyfert2 | starforming | type1 | type1_host | unknown |
|---|---|---|---|---|---|---|---|---|---|
| comp | 72 | 6 | 2 | 3 | 13 | 3 | 0 | 1 | 0 |
| liner | 3 | 38 | 37 | 37 | 35 | 0 | 0 | 0 | 0 |
| passive | 0 | 0 | 92 | 8 | 0 | 0 | 0 | 0 | 0 |
| sf | 12 | 1 | 2 | 0 | 2 | 130 | 0 | 0 | 3 |
| sy2 | 2 | 7 | 8 | 4 | 127 | 0 | 0 | 2 | 0 |
| t1_low | 16 | 2 | 9 | 0 | 63 | 50 | 91 | 51 | 18 |
