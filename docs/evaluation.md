# Evaluation

Everything here is measured on synthetic airports generated with seeds the demo does not use (`(101, 102, 103)`), on 3 storm
days each, with the storm starting between 14:00 and 18:00 and the clock set to its start. Reproduce with `python -m ap.evaluate`
(about 5 minutes). The numbers say how well the methods do against the simulator's hidden version of each day. They are not a
claim about a real airport: real delays have causes this generator does not have, and real gate rules are richer.

## 1. How late will each aircraft be?

Trained on 72 past days of the same timetable, tested on 18 later days, each flight seen from a random 30 to 180 minutes before
its scheduled landing. The baseline is what a departures board shows: the delay the aircraft left its origin with, or on time
if it has not left yet.

| airport (seed) | test flights | error, model | error, board | in a storm queue, model | in a storm queue, board | inside the 10–90% band |
|---|---|---|---|---|---|---|
| 101 | 5238 | 4.24 | 4.52 | 4.29 | 5.1 | 80.1% |
| 102 | 5328 | 4.23 | 4.54 | 4.98 | 5.84 | 78.9% |
| 103 | 5184 | 4.87 | 4.7 | 7.45 | 6.36 | 79.7% |

Errors are mean absolute errors in minutes. The model's gain is small on an ordinary day and clearer in a storm queue, where the
board knows nothing about the runway. 80% of outcomes inside the 10–90% band would be perfect calibration.

## 2. Is the chance of a missed connection honest?

Every connecting group on the 9 evaluation evenings whose inbound aircraft was still to arrive, under the scheduled gates:
the chance the system gave against the share that really missed (passenger-weighted). Where the two columns differ, the system
is too sure of itself at the top end: a "90%+" group misses less often than that, because the departure it is running for is
often late too.

| predicted chance | groups | mean predicted | really missed |
|---|---|---|---|
| 0.0 to 0.1 | 6359 | 0.9% | 1.9% |
| 0.1 to 0.3 | 141 | 18.5% | 21.4% |
| 0.3 to 0.5 | 74 | 39.3% | 35.2% |
| 0.5 to 0.7 | 73 | 60.9% | 57.5% |
| 0.7 to 0.9 | 121 | 78.4% | 58.7% |
| 0.9 to 1.0 | 213 | 98.4% | 85.9% |

## 3. Do the plans help?

The rest of each day replayed as it really went, counting aircraft that arrived after the storm began. "First free gate" is
the obvious rule: take the nearest free compatible gate when you arrive. "Holds" wait up to 15 minutes for connecting passengers
whose aircraft is already at a gate. "New gates" is this system's plan, judged on six sampled futures. "Perfect knowledge" plans
with the real arrival times, which nothing could have.

| plan | gate waiting (aircraft-min) | missed connections | departures held (min) | gate changes | cost (passenger-min) |
|---|---|---|---|---|---|
| scheduled gates | 14825 | 2796 | 0 (0) | 0 | 5,473,473 |
| first free gate | 13528 | 3190 | 0 (0) | 530 | 5,835,732 |
| scheduled gates + holds | 14822 | 2148 | 141 (673) | 0 | 5,423,580 |
| new gates + holds | 15551 | 2212 | 130 (622) | 88 | 5,237,693 |
| perfect knowledge + holds | 15955 | 2168 | 125 (554) | 122 | 5,097,668 |

Totals over the 9 evenings. By evening (missed connections, scheduled gates / first free / scheduled + holds / new gates + holds / perfect):

| airport | day | scheduled | first free | + holds | new gates + holds | perfect + holds |
|---|---|---|---|---|---|---|
| 101 | 0 | 289 | 293 | 247 | 247 | 186 |
| 101 | 1 | 178 | 345 | 108 | 203 | 188 |
| 101 | 2 | 97 | 124 | 80 | 80 | 80 |
| 102 | 0 | 411 | 459 | 293 | 297 | 348 |
| 102 | 1 | 243 | 318 | 190 | 189 | 202 |
| 102 | 2 | 304 | 333 | 264 | 255 | 273 |
| 103 | 0 | 347 | 366 | 271 | 216 | 217 |
| 103 | 1 | 609 | 567 | 483 | 513 | 464 |
| 103 | 2 | 318 | 385 | 212 | 212 | 210 |

What the table says. Holding departures cuts missed connections by 23.2% at a price of
673 minutes of held departures. The new gates lower the passenger-minute cost by 4.3% (perfect knowledge
of arrival times: 6.9%), mostly by keeping full aircraft off the taxiway, but they do not cut missed connections
further: the optimiser minimises the cost it is given, in which a missed connection is worth 90 passenger-minutes, and it
trades some of them for waiting time. In a storm the gates are full, so any assignment mostly decides who waits. The cost
column is dominated by departure delay the storm itself causes, which no plan can remove.
