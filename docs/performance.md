# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 292 flights, 2,644 connecting groups, two plans.
Reproduce with `make demo`, then `make load-test`.

## Reads: operations view (6 h and 2 h windows) and one flight's connection risk, in rotation, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 448 | 44.8 req/s | 11.1 ms | 17.4 ms | 24.1 ms | 0.00% |
| 8 | 1135 | 112.9 req/s | 67.0 ms | 93.5 ms | 103.7 ms | 0.00% |
| 32 | 979 | 95.7 req/s | 269.9 ms | 333.2 ms | 376.9 ms | 0.00% |

Target from the blueprint: p95 under 500 ms for reads: met. The operations view re-runs the gate simulator and prices every
connection; results are cached per state of knowledge (what is known about each flight, the storm, the published plan), so a
burst of reads between events costs one computation. Before that cache the p95 at 32 concurrent readers was 2.5 s.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Generate the airport: timetable, gate plan, 2,644 connecting groups | 0.04 s |
| Train the delay model on 21,024 past flights (three gradient-boosted fits), test on 5,256 | 1.7 s |
| One replay of the evening through the gate simulator | 3 ms |
| Propose a plan: 800 candidates on 6 sampled futures | 12 s |
| Evaluation: three airports, nine storm evenings, four plans each | about 2 minutes |

Not measured: a live event stream. The demo stands at 16:00; with time moving, estimates would be recomputed per event and
the cache would turn over at that rate. The blueprint's scale (a day of thousands of flights, sub-minute telemetry) is beyond
this slice.
