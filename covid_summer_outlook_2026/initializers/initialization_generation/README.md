## For instructions return to [covid_summer_outlook_2026](/covid_summer_outlook_2026//README.md)

### Folder Content
- `bin_tslie.py`: distributes population infection history into SEIS compartments.
- `formatting_data.py`: formats raw data into format needed for initialization code.
- `import_data.py`: saves raw data needed to run initialization code and removed data of unused dates.
- `initialization_config.json`: contains necessary variables to run initialization code base.
- `process_data.py`: apply any functions necessary to data such as scaling or normalizing, and combining data sets to estimate individuals that have combination of different datas.
- `run.py` : function that generates initialization and uses all other functions in repo.
- `tslie.py` : calculates time since last immunization event from processed data for estimated individuals.
- `validate_processed_data.py` : basic validators to confirm processing of data doesn't lead to impossible behaviors like negative individuals or more infections than total people.
- `validate_raw_data.py` : basic validators to confirm raw data doesn't contain impossible behaviors like negative individuals or more infections than total people.
- `validate_tslie.py` : basic validators to confirm tslie.py doesn't lead to impossible behaviors like negative individuals or more infections than total people.
- **`data/`**: model inputs for demographics, hospitalization,
vaccination, and variant proportions
