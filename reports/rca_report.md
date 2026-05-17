# Root Cause Analysis Report
**Project:** AI-based 5G Alarm Analysis — Djezzy  
**Phase:** 03 — Root Cause Analysis  

---
## 1. Top Root Causes per HS Category

### 3G cell down
| Rank | Alarm | Composite Score |
|------|-------|-----------------|
| 1 | Adjacent Node IP Address Ping Failure | 0.664 |
| 2 | SCTP Link Fault | 0.499 |
| 3 | 3G PS Call Setup Success Rate | 0.236 |
| 4 | 3G CS Call Drop Rate | 0.231 |
| 5 | UMTS Cell MC-HSDPA Function Fault | 0.174 |

### LOSS-OF-ALL CHANNEL
| Rank | Alarm | Composite Score |
|------|-------|-----------------|
| 1 | Adjacent Node IP Address Ping Failure | 0.379 |
| 2 | SCTP Link Fault | 0.218 |
| 3 | 3G PS Call Setup Success Rate | 0.217 |
| 4 | 3G CS Call Drop Rate | 0.216 |
| 5 | RF Unit Maintenance Link Failure | 0.166 |

### Pas De Supervision
| Rank | Alarm | Composite Score |
|------|-------|-----------------|
| 1 | The Data Transmission Channel Between the Trace Server and the NE Is Disrupted | 0.364 |
| 2 | Cell Outage | 0.251 |
| 3 | Parallel Alarm Exceeds the Limit | 0.166 |
| 4 | Task execution failure alarm | 0.066 |
| 5 | The Number of Login Attempts Reaches the Maximum | 0.008 |

### 2G cell down
| Rank | Alarm | Composite Score |
|------|-------|-----------------|
| 1 | Adjacent Node IP Address Ping Failure | 0.344 |
| 2 | SCTP Link Fault | 0.200 |
| 3 | RF Unit Maintenance Link Failure | 0.192 |
| 4 | Commercial Power Down (Main Power Down) | 0.174 |
| 5 | Local Cell Unusable | 0.151 |

### 4G cell down
| Rank | Alarm | Composite Score |
|------|-------|-----------------|
| 1 | RF Unit Maintenance Link Failure | 0.230 |
| 2 | Local Cell Unusable | 0.196 |
| 3 | X2 Interface Fault | 0.141 |
| 4 | SCTP Link Fault | 0.135 |
| 5 | User Plane Fault | 0.132 |

---
## 2. Causal Chains
| Root Alarm | Intermediate | Terminal (HS) | Corr | Lag (h) |
|------------|-------------|--------------|------|----------|
| RF Unit RX Channel RTWP/RSSI Unbalanced | SCTP Link Fault | 2G cell down | 0.192 | 1 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | SCTP Link Fault | 2G cell down | 0.191 | 3 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | SCTP Link Fault | 2G cell down | 0.189 | 2 |
| — | ESL Link Fault | 2G cell down | — | — |
| — | E1/T1 Alarm Indication Signal | 2G cell down | — | — |
| RF Unit RX Channel RTWP/RSSI Unbalanced | Adjacent Node IP Address Ping Failure | 2G cell down | 0.277 | 1 |
| X2 Interface Fault | Adjacent Node IP Address Ping Failure | 2G cell down | 0.270 | 1 |
| Cell RX Channel Interference Noise Power Unbalanced | Adjacent Node IP Address Ping Failure | 2G cell down | 0.265 | 1 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | Commercial Power Down (Main Power Down) | 2G cell down | 0.419 | 1 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | Commercial Power Down (Main Power Down) | 2G cell down | 0.396 | 2 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | Commercial Power Down (Main Power Down) | 2G cell down | 0.384 | 3 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | SCTP Link Fault | 3G cell down | 0.192 | 1 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | SCTP Link Fault | 3G cell down | 0.191 | 3 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | SCTP Link Fault | 3G cell down | 0.189 | 2 |
| RF Unit RX Channel RTWP/RSSI Unbalanced | Adjacent Node IP Address Ping Failure | 3G cell down | 0.277 | 1 |
| X2 Interface Fault | Adjacent Node IP Address Ping Failure | 3G cell down | 0.270 | 1 |
| Cell RX Channel Interference Noise Power Unbalanced | Adjacent Node IP Address Ping Failure | 3G cell down | 0.265 | 1 |
| Cell RX Channel Interference Noise Power Unbalanced | UMTS Cell MC-HSDPA Function Fault | 3G cell down | 0.253 | 4 |
| Cell RX Channel Interference Noise Power Unbalanced | UMTS Cell MC-HSDPA Function Fault | 3G cell down | 0.251 | 5 |
| Cell RX Channel Interference Noise Power Unbalanced | UMTS Cell MC-HSDPA Function Fault | 3G cell down | 0.249 | 6 |

---
## 3. High-Risk Sites
| Rank | Site | HS Count | HS Rate | Risk Score |
|------|------|----------|---------|------------|
| 1 | ROR03 | 19102 | 34.86% | 0.837 |
| 2 | RAL03 | 10984 | 24.81% | 0.585 |
| 3 | RCN03 | 9589 | 33.48% | 0.582 |
| 4 | Annaba RNC | 13485 | 12.70% | 0.503 |
| 5 | RLI03 | 9154 | 12.24% | 0.488 |
| 6 | UNC02_Annaba | 5726 | 76.39% | 0.483 |
| 7 | OSS | 14461 | 13.01% | 0.463 |
| 8 | UNC02_Tlemcen | 5394 | 52.13% | 0.379 |
| 9 | BA06H | 398 | 62.09% | 0.354 |
| 10 | BA02H | 529 | 59.24% | 0.346 |
