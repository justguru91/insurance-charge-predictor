# Dataset source and attribution

Dataset: **Medical Cost Personal / insurance.csv**.

Downloaded on 24 October 2026 from the public repository of formatted datasets for Brett Lantz's *Machine Learning with R*, maintained by GitHub user **stedy**:

- Repository: https://github.com/stedy/Machine-Learning-with-R-datasets
- Exact source page: https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv
- Download URL: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
- Related Kaggle listing by Miri Choi: https://www.kaggle.com/datasets/mirichoi0218/insurance

The untouched downloaded file is `raw.csv`. Its SHA-256 fingerprint is `505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281`. The fingerprint is also written to the cleaning audit and checked by `src/verify.py`.

The seven columns are age, sex, bmi, children, smoker, region, and charges. The first six are model inputs; charges is the target. Region categories refer to US regions. The project treats charge values as US dollars, following common use of this benchmark. The source does not establish a current tariff, a verified policy period, or representative coverage of another country. Do not present these numbers as live Indian premiums or quotes.

This project does not claim to have collected these records or verified their original collection process. Preserve this attribution with academic submissions. Dataset source terms remain separate from the project code.
