# Adversarial disentanglement runs summary

Learned routing created a moving target: while the adversary was trying to clean speaker information from `z_L`, units could keep changing route assignment. These runs therefore test fixed-block alternatives, GRL strength/schedules, U-buffer variants, and a learned-then-frozen routing control.

Training structure, once for all runs: stage-2 routed SAE with `K=5120`, `topk=256`; reconstruction + PR + SID auxiliaries; speaker adversary on `z_L` through a linear-mean GRL head; phoneme adversary on `z_P`; final-checkpoint probes are fresh diagnostic heads. PR entries below report the best reliable PR probe PER for each source after dropping clear probe-initialization failures. SID reports linear and stats for `z_L/z_P`; for `z_t`, only linear SID is reported.

## Fixed-block allocation controls

Motive: test whether changing the fixed active split changes the PR/cleaning trade-off.

| Run | Active L/P/U | Val recon loss | Dead | ckpt PR PER | PR PER z_t / z_L / z_P | z_t SID | z_L SID lin/stat | z_P SID lin/stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Colab fixed `224L/32P`, `gn=2.5e-4` | 224/32/0 | 0.1925 | 15.4% | — | 18.3% / 19.7% / 99.2% | 99.4% | 0.2% / partial | 100.0% / — |
| Blackwell fixed `240L/16P`, `gn=2.5e-4` | 240/16/0 | 0.1891 | 13.5% | 15.1% | — / 16.4% / — | — | — | — |

## GRL strength controls

Motive: find how low the `z_L` speaker-cleaning pressure can go before speaker information leaks back into `z_L`.

| Run | Active L/P/U | Val recon loss | Dead | ckpt PR PER | PR PER z_t / z_L / z_P | z_t SID | z_L SID lin/stat | z_P SID lin/stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant `gn=1.5e-4` | 240/16/0 | 0.1787 | 8.9% | 9.5% | 9.8% / 10.4% / 100.0% | 100.0% | 2.6% / 0.2% | 99.8% / 100.0% |
| Constant `gn=1e-4` | 240/16/0 | 0.1598 | 5.9% | 7.3% | 7.6% / 7.8% / 100.0% | 100.0% | 41.8% / 0.2% | 99.6% / 100.0% |

## GRL schedule controls

Motive: clean speaker early, then relax adversarial pressure to see if PR recovers without speaker leakage.

| Run | Active L/P/U | Val recon loss | Dead | ckpt PR PER | PR PER z_t / z_L / z_P | z_t SID | z_L SID lin/stat | z_P SID lin/stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Late decay `2.5e-4→5e-5` | 240/16/0 | 0.1850 | 12.3% | 13.3% | 11.4% / 14.7% / 100.0% | 99.6% | 1.4% / 0.2% | 99.8% / 100.0% |
| Early decay `2e-4→5e-5` | 240/16/0 | 0.1794 | 9.1% | 10.6% | failed / 10.7% / 100.0% | 100.0% | 1.4% / 0.2% | 99.6% / 100.0% |

## U-block controls

Motive: test whether a small U block can absorb leftover information without corrupting the L/P split.

| Run | Active L/P/U | Val recon loss | Dead | ckpt PR PER | PR PER z_t / z_L / z_P / z_U | z_t SID | z_L SID lin/stat | z_P SID lin/stat | z_U SID lin/stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U24 + U adversaries | 216/16/24 | 0.1941 | 20.8% | 18.0% | 22.5% / 19.9% / 99.8% / 100.0% | 99.6% | 0.4% / 0.2% | 100.0% / 99.6% | 0.4% / 0.4% |
| U24 open | 216/16/24 | 0.1411 | 11.7% | 16.7% | 11.1% / 11.8% / 99.9% / 27.7% | 100.0% | 1.4% / 0.2% | 99.2% / 99.6% | 96.6% / 92.0% |

## Learned-routing dynamics control

Motive: test whether learned routing failed because routes kept moving during adversarial cleaning.

| Run | Final units L/P/U | Active L/P/U | Val recon loss | Dead | ckpt PR PER | PR PER z_t / z_L / z_P | z_t SID | z_L SID lin/stat | z_P SID lin/stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard learned → freeze@4k | 1791/3329/0 | 240/16/0 | 0.1411 | 0.2% | 6.4% | 6.6% / 7.0% / 92.5% | 99.4% | 93.0% / incomplete | 99.8% / incomplete |

## Short conclusion

The most balanced fixed-block setting is currently `gn=1.5e-4`: `z_L` PR is about 10.4% PER while linear SID leakage stays low at 2.6%. Lower `gn=1e-4` and learned-freeze recover much better PR, but both leak too much speaker information into `z_L`. Open U helps reconstruction but becomes a speaker reservoir; adversarial U stays clean but damages PR/reconstruction and increases dead units.
