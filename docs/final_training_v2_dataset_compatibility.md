# Final CHB-MIT V2 dataset compatibility

The canonical model input is the established 23-channel order. Extra channels are safely excluded before Batch-A preprocessing; missing canonical channels are never synthesized.

- EDFs inspected: 683
- Canonical-23 usable EDFs: 652
- Excluded EDFs: 31
- Safe extra-channel mappings: 379

## Safe additional-channel conversion

379 recordings contain all canonical channels plus additional channels. The pipeline deterministically selects the canonical channels in model order before filtering/CAR; no data channels are fabricated.

## Exclusions

### missing_canonical_channels

Count: 31

Safe conversion: no. One or more canonical model channels are absent.

Representative recordings:
- `chb12/chb12_27` — missing canonical channels: FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8-0, P8-O2, FZ-CZ, CZ-PZ, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb12/chb12_28` — missing canonical channels: FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8-0, P8-O2, FZ-CZ, CZ-PZ, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb12/chb12_29` — missing canonical channels: FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8-0, P8-O2, FZ-CZ, CZ-PZ, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb13/chb13_04` — missing canonical channels: T8-P8-0, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb13/chb13_05` — missing canonical channels: T8-P8-0, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb13/chb13_06` — missing canonical channels: T8-P8-0, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb13/chb13_07` — missing canonical channels: T8-P8-0, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb13/chb13_08` — missing canonical channels: T8-P8-0, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb13/chb13_09` — missing canonical channels: T8-P8-0, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- `chb13/chb13_10` — missing canonical channels: T8-P8-0, P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
