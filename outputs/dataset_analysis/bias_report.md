# WikiArt Dataset Label Audit

- Dataset: `huggan/wikiart`
- Split: `train`

## Bias Summary

### full_dataset

- Total rows processed: `81444`

#### Artist

- Total labels: `129`
- Labels with at least one sample: `129`
- Largest class: `Unknown Artist` with `41914` works (`51.4636%`)
- Smallest non-zero class: `canaletto` with `42` works
- Top 10 classes share: `64.4455%`
- Labels with 1 sample: `0`
- Labels with fewer than 10 samples: `0`
- Labels with fewer than 50 samples: `2`

#### Genre

- Total labels: `11`
- Labels with at least one sample: `11`
- Largest class: `Unknown Genre` with `16452` works (`20.2004%`)
- Smallest non-zero class: `illustration` with `1902` works
- Top 10 classes share: `97.6647%`
- Labels with 1 sample: `0`
- Labels with fewer than 10 samples: `0`
- Labels with fewer than 50 samples: `0`

#### Style

- Total labels: `27`
- Labels with at least one sample: `27`
- Largest class: `Impressionism` with `13060` works (`16.0356%`)
- Smallest non-zero class: `Action_painting` with `98` works
- Top 10 classes share: `76.6588%`
- Labels with 1 sample: `0`
- Labels with fewer than 10 samples: `0`
- Labels with fewer than 50 samples: `0`

### known_artist_only

- Total rows processed: `39530`

#### Artist

- Total labels: `129`
- Labels with at least one sample: `128`
- Largest class: `vincent-van-gogh` with `1889` works (`4.7786%`)
- Smallest non-zero class: `canaletto` with `42` works
- Top 10 classes share: `28.6744%`
- Labels with 1 sample: `0`
- Labels with fewer than 10 samples: `1`
- Labels with fewer than 50 samples: `3`

#### Genre

- Total labels: `11`
- Labels with at least one sample: `11`
- Largest class: `portrait` with `8561` works (`21.657%`)
- Smallest non-zero class: `abstract_painting` with `501` works
- Top 10 classes share: `98.7326%`
- Labels with 1 sample: `0`
- Labels with fewer than 10 samples: `0`
- Labels with fewer than 50 samples: `0`

#### Style

- Total labels: `27`
- Labels with at least one sample: `25`
- Largest class: `Impressionism` with `9202` works (`23.2785%`)
- Smallest non-zero class: `Minimalism` with `5` works
- Top 10 classes share: `88.9831%`
- Labels with 1 sample: `0`
- Labels with fewer than 10 samples: `3`
- Labels with fewer than 50 samples: `4`
