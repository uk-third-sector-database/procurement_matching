import os
import re
import html
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from matching_helpers import normaliser,\
                             parse_datetime,\
                             read_raw_data,\
                             prepare_nhsspend,\
                             prepare_contractsfinder,\
                             org_counter,\
                             strip_html,\
                             unique_agg,\
                             process_dates,\
                             make_matches,\
                             filter_coercible_to_string


def main():
    df_centgov = read_raw_data('centgov_data.csv', 'Contracts Finder')
    df_nhs = read_raw_data('nhsspend_data.csv', 'NHSSpend')
    df_contracts = read_raw_data('contractsfinder_data.csv', 'Contracts Finder')
    df_centgov = df_centgov[['data_source', 'amount', 'supplier', 'date', 'dept']]
    df_nhs = prepare_nhsspend(df_nhs)
    df_contracts = prepare_contractsfinder(df_contracts)

    df_comb = pd.concat([df_nhs, df_centgov, df_contracts], ignore_index=True)
    df_comb = df_comb.rename({'supplier': 'SUPPLIER'}, axis=1)
    df_comb['SUPPLIER'] = df_comb['SUPPLIER'].str.upper().str.strip()

    df_comb['SUPPLIER_NUMERIC'] = pd.to_numeric(df_comb['SUPPLIER'], errors='coerce')
    print(f'Dropping {len(df_comb[df_comb["SUPPLIER"].isnull()])} rows of data because of numeric suppliers')
    df_comb = df_comb[df_comb['SUPPLIER'].notnull()]
    df_comb = df_comb[df_comb['SUPPLIER_NUMERIC'].isna()]
    df_comb = df_comb.drop(columns='SUPPLIER_NUMERIC')

    print(f'Dropping {len(df_comb[df_comb["SUPPLIER"].isnull()])} rows of data because of NaN suppliers')
    df_comb = df_comb[df_comb['SUPPLIER'].notnull()]
    df_comb['SUPPLIER'] = df_comb['SUPPLIER'].progress_apply(strip_html)
    print(f'Dropping {len(df_comb[df_comb["SUPPLIER"].isnull()])} rows of data after html parsing')
    df_comb = df_comb[df_comb['SUPPLIER'].notnull()]
    df_comb['date'] = df_comb['date'].astype(str).str.split('T').str[0]
    df_comb['date'] = pd.to_datetime(df_comb['date'],
                                     format='mixed',
                                     errors='coerce')
    df_comb['date'] = df_comb['date'].map(lambda x: x.strftime('%d-%m-%Y') if pd.notnull(x) else np.nan)
    print(f'Dropping {len(df_comb[df_comb["date"].isnull()])} rows of data due to NaN dates')
    df_comb = df_comb[df_comb['date'].notnull()]
    print(f'Dropping {len(df_comb[df_comb["amount"].isnull()])} rows of data due to NaN amounts')
    df_comb = df_comb[df_comb['amount'].notnull()]
    print(f'Dropping {len(df_comb[df_comb["dept"].isnull()])} rows of data due to NaN depts')
    df_comb = df_comb[df_comb['dept'].notnull()]
    df_comb['NORMALIZED_SUPPLIER'] = df_comb['SUPPLIER'].progress_apply(normaliser)

    rows_to_drop = len(df_comb[
                           (df_comb["SUPPLIER"].str.len() <= 3) |
                           (df_comb["NORMALIZED_SUPPLIER"].str.len() <= 3)
                           ])

    # Print the message with the count of rows to be dropped
    print(f'Dropping {rows_to_drop} rows of data due to supplier str len<=3')

    df_comb = df_comb[
        (df_comb["SUPPLIER"].str.len() > 3) |
        (df_comb["NORMALIZED_SUPPLIER"].str.len() > 3)
        ]

    df_comb[['SUPPLIER', 'ORG_COUNT']] = df_comb['SUPPLIER'].apply(lambda x: pd.Series(org_counter(x)))

    all_rows = len(df_comb)

    for supplier in ["SUCCESSFUL SUPPL",
                     "SEE ATTACH",
                     "REFER ATTACH",
                     "CONTRACT WAS AWARD",
                     "AWARDED SUPPLIERS",
                     "SUCCESSFUL SUPPLIER",
                     "PLEASE SEE",
                     'NAMED IND',
                     'REDACT',
                     "PLEASE REFER"]:
        df_comb = df_comb[~df_comb['SUPPLIER'].str.contains(supplier)]

    print(f'Number of rows dropped due to redacted: {len(df_comb) - all_rows}')

    print(f'Dropping {len(df_comb[df_comb["ORG_COUNT"] != 1])} where org_count !=1')
    df_comb = df_comb[df_comb['ORG_COUNT'] == 1]

    df_uniq = df_comb.pivot_table(index=['SUPPLIER'],
                                  values=['date',
                                          'contractsfinder_region',
                                          'contractsfinder_awardedToVcse',
                                          'dept'],
                                  aggfunc=unique_agg).reset_index()
    df_sum = df_comb.groupby('SUPPLIER')['amount'].sum().reset_index()
    df_counts = df_comb['SUPPLIER'].value_counts().reset_index()
    df_uniq = pd.merge(df_uniq,
                       df_sum,
                       how='left',
                       left_on='SUPPLIER',
                       right_on='SUPPLIER'
                       )
    df_uniq[['SUPPLIER', 'ORG_COUNT']] = df_uniq['SUPPLIER'].progress_apply(lambda x: pd.Series(org_counter(x)))
    df_uniq['NORMALIZED_SUPPLIER'] = df_uniq['SUPPLIER'].progress_apply(normaliser)
    df_uniq = pd.merge(df_uniq,
                       df_counts,
                       how='left',
                       left_on='SUPPLIER',
                       right_on='SUPPLIER'
                       )
    df_uniq.sort_values(by=['amount'],
                        ascending=False)

    df_uniq1 = df_nhs[['supplier',
                       'NHSSpend_CompanyName',
                       'NHSSpend_CompanyNumber',
                       'NHSSpend_CharityName',
                       'NHSSpend_CharityRegNo',
                       'NHSSpend_CharitySubNo',
                       'NHSSpend_CharityNameNo',
                       'NHSSpend_CharityName']].drop_duplicates()
    df_uniq = pd.merge(df_uniq,
                       df_uniq1,
                       how='left',
                       left_on='SUPPLIER',
                       right_on='supplier'
                       )

    contractsfinder_region = df_sum = df_comb.groupby('SUPPLIER')['amount'].sum().reset_index()

    df_uniq = df_uniq.rename({'count': 'PAYMENT_TOTAL_COUNT',
                              'amount': 'PAYMENT_TOTAL_AMOUNT'},
                             axis=1)
    print(f'Dropping {len(df_uniq[df_uniq["ORG_COUNT"] != 1])} org_count !=1')
    df_uniq = df_uniq[df_uniq['ORG_COUNT'] == 1]
    df_uniq = df_uniq.drop(columns='supplier')
    df_uniq['contractsfinder_awardedToVcse'] = df_uniq['contractsfinder_awardedToVcse'].apply(
        lambda x: "True" if True in x else "False")
    df_uniq['deptcount'] = df_uniq['dept'].astype(str).apply(lambda x: x.count(';') + 1)
    df_uniq['date'] = df_uniq['date'].astype(str).progress_apply(process_dates)
    df_uniq['SUPPLIER'] = df_uniq['SUPPLIER'].replace('"', "[DQ]", regex=True)

    df_uniq = df_uniq.drop('NHSSpend_CharityNameNo', axis=1)
    df_uniq = df_uniq.drop('NHSSpend_CharityName', axis=1)
    df_uniq = df_uniq.drop('NHSSpend_CompanyName', axis=1)

    df_comb = df_comb.drop('NHSSpend_CharityNameNo', axis=1)
    df_comb = df_comb.drop('NHSSpend_CharityName', axis=1)
    df_comb = df_comb.drop('NHSSpend_CompanyName', axis=1)
    df_comb = df_comb.drop('NHSSpend_audit_type', axis=1)
    df_comb = df_comb.drop('NHSSpend_CHnotes', axis=1)
    df_comb = df_comb.drop('NHSSpend_CCnotes', axis=1)

    print(df_uniq.columns)
    print(df_uniq.head())

    print(df_comb.columns)
    print(df_comb.head())

    df_uniq = df_uniq.sort_values(by='PAYMENT_TOTAL_AMOUNT',
                                  ascending=False)
    df_uniq['NORMALIZED_SUPPLIER'] = filter_coercible_to_string(df_uniq['NORMALIZED_SUPPLIER'])
    df_uniq.to_csv(os.path.join(os.getcwd(),
                                '..',
                                'raw_data',
                                'merged_groupby_raw.csv'),
                   index=False)

    df_comb.to_csv(os.path.join(os.getcwd(),
                                '..',
                                'raw_data',
                                'merged_all_raw.csv'),
                   index=False
                   )

    print(f'We are then left with {len(df_uniq)} rows of unique "single" suppliers')
    print(f'We are then left with {len(df_comb)} rows of unique "single" payments')

    df_ch = pd.read_csv(os.path.join(
        '..', 'registers', 'BasicCompanyDataAsOneFile-2024-08-01.csv'),
        usecols=['CompanyName', ' CompanyNumber', 'RegAddress.PostTown', 'RegAddress.PostCode']
    )
    df_ch['NORMALIZED_CompanyName'] = df_ch['CompanyName'].astype(str).progress_apply(normaliser)
    df_ch['NORMALIZED_CompanyName'] = filter_coercible_to_string(df_ch['NORMALIZED_CompanyName'])
    df_ch = df_ch.drop_duplicates(subset=['NORMALIZED_CompanyName'], keep=False)
    df_ch = df_ch[
        [' CompanyNumber', 'NORMALIZED_CompanyName', 'CompanyName', 'RegAddress.PostTown', 'RegAddress.PostCode']]
    df_ch.to_csv(os.path.join('..', 'registers', 'ch_w_normalised.csv'), index=False)

    df_spine = pd.read_csv(os.path.join(
        '..', 'registers', 'public_spine.spine.csv'),
        usecols=['uid', 'organisationname', 'fulladdress', 'city', 'postcode', 'registerdate', 'removeddate']
    )
    df_spine['NORMALIZED_organisationname'] = df_spine['organisationname'].astype(str).progress_apply(normaliser)
    df_spine['NORMALIZED_organisationname'] = filter_coercible_to_string(df_spine['NORMALIZED_organisationname'])
    df_spine = df_spine.drop_duplicates(subset=['NORMALIZED_organisationname'], keep=False)
    df_spine = df_spine[
        ['uid', 'NORMALIZED_organisationname', 'organisationname', 'fulladdress', 'city', 'postcode', 'registerdate',
         'removeddate']]
    df_spine.to_csv(os.path.join('..', 'registers', 'spine_w_normalised.csv'), index=False)

    print('Beginning to make the spine matches')
    df_spine_results = make_matches(df_uniq['NORMALIZED_SUPPLIER'], df_spine['NORMALIZED_organisationname'], 'spine')
    df_spine_results.to_csv(os.path.join('..', 'matches', 'matches_to_spine.csv'))

    df_uniq['verified_normalized_spine_name'] = np.nan
    df_uniq['verified_spine_uid'] = np.nan
    df_uniq['verified_normalized_ch_name'] = np.nan
    df_uniq['verified_ch_uid'] = np.nan
    df_uniq = df_uniq.join(df_spine_results, how='left')
    df_uniq.to_csv(os.path.join(os.getcwd(),
                                '..',
                                'raw_data',
                                'merged_groupby_with_approximate_spine.csv'),
                   index=False)

    print('Beginning to make the CH matches')
    df_ch_results = make_matches(df_uniq['NORMALIZED_SUPPLIER'], df_ch['NORMALIZED_CompanyName'], 'ch')
    df_ch_results.to_csv(os.path.join('..', 'matches', 'matches_to_ch.csv'))
    df_uniq = df_uniq.join(df_ch_results, how='left')
    df_uniq.to_csv(os.path.join(os.getcwd(),
                               '..',
                               'raw_data',
                               'merged_groupby_with_approximate_spine_and_ch.csv'),
                   index=False)


if __name__ == "__main__":
    main()