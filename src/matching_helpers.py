import os
import re
import warnings
import pandas as pd
from bs4 import BeautifulSoup
from thefuzz import process as thefuzz_process
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

def fuzzy_match(supplier, choices):
    return thefuzz_process.extract(supplier, choices=choices, limit=5)

def is_string_coercible(value):
    """Helper function to check if a value can be coerced to a string."""
    try:
        str(value)
        return True
    except (ValueError, TypeError):
        return False

def filter_coercible_to_string(series):
    """Filter out any values in the pandas Series that cannot be coerced to a string."""
    return series[series.apply(is_string_coercible)].astype(str)

def make_matches(input_1, input_2, match_type):
    suppliers = input_1.tolist()
    choices = input_2.tolist()
    results = Parallel(n_jobs=36)(
        delayed(fuzzy_match)(supplier, choices) for supplier in tqdm(suppliers)
    )
    columns = [
        "best_" + match_type + "_match_1", "best_" + match_type + "_match_1_score",
        "best_" + match_type + "_match_2", "best_" + match_type + "_match_2_score",
        "best_" + match_type + "_match_3", "best_" + match_type + "_match_3_score",
        "best_" + match_type + "_match_4", "best_" + match_type + "_match_4_score",
        "best_" + match_type + "_match_5", "best_" + match_type + "_match_5_score"
    ]
    rows = []
    for match_list in results:
        row = []
        for match, score in match_list:
            row.extend([match, score])
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def process_dates(date_string):
    date_string = date_string.strip("[]").replace("'", "")
    dates = pd.to_datetime([d.strip() for d in date_string.split(',')], format="%d-%m-%Y", errors='coerce')
    dates = dates.dropna()
    if len(dates) > 1:
        return f"{dates.min().strftime('%d-%m-%Y')}-{dates.max().strftime('%d-%m-%Y')}"
    elif len(dates) == 1:
        return dates[0].strftime('%d-%m-%Y')
    else:
        return ""


def unique_agg(series):
    return list(series.unique())


def strip_html(html):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


def org_counter(supplier):
    if ',' not in supplier:
        return supplier.upper(), 1
    parts = supplier.split(',')
    processed_parts = [part.strip().upper() for part in parts if len(part.strip()) >= 4]
    unique_parts = set(processed_parts)
    if len(unique_parts) == 1:
        return unique_parts.pop(), 1
    else:
        return supplier, len(unique_parts)


def prepare_contractsfinder(df_contracts):
    df_contracts['date'] = df_contracts['awardedDate']
    df_contracts['supplier'] = df_contracts['awardedSupplier']
    df_contracts['amount'] = df_contracts['awardedValue']
    df_contracts['dept'] = df_contracts['organisationName']
    df_contracts['contractsfinder_awardedToVcse'] = df_contracts['awardedToVcse']
    df_contracts['contractsfinder_region'] = df_contracts['region']
    return df_contracts[['data_source',
                         'amount',
                         'supplier',
                         'date',
                         'dept',
                         'contractsfinder_awardedToVcse',
                         'contractsfinder_region'
                        ]]

def prepare_nhsspend(df_nhs):
    for col in ['CompanyName', 'CompanyNumber', 'CharityRegNo',
                'CharitySubNo', 'CharityNameNo','CharityName',
                'audit_type', 'CHnotes', 'CCnotes', 'isCIC']:
        df_nhs = df_nhs.rename({col: 'NHSSpend_'+col}, axis=1)
    return df_nhs[['data_source', 'amount', 'supplier', 'date', 'dept',
                   'NHSSpend_CompanyName',
                   'NHSSpend_CompanyNumber',
                   'NHSSpend_CharityRegNo',
                   'NHSSpend_CharitySubNo',
                   'NHSSpend_CharityNameNo',
                   'NHSSpend_CharityName',
                   'NHSSpend_audit_type',
                   'NHSSpend_CHnotes',
                   'NHSSpend_CCnotes',
                   'NHSSpend_isCIC']]


def parse_datetime(value):
    try:
        return pd.to_datetime(value, format='%Y/%m/%d', errors='coerce')
    except:
        return pd.NaT


def read_raw_data(fname, data_type):
    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'raw_data',
                                  fname),
                     low_memory=False,
                    )
    df['data_source'] = data_type
    return df


def process_cleanname(cleanname):
    cleanname = " " + cleanname + " "
    cleanname = cleanname.replace(" PTFA ", " PTA ")
    cleanname = cleanname.replace(" PSA ", " PTA ")
    if re.search(r" PARENT", cleanname) and re.search(r" TEACHER", cleanname) and re.search(r"ASSOC ", cleanname):
        cleanname += " PTA "
    if re.search(r" PARENT", cleanname) and re.search(r" STAFF", cleanname) and re.search(r"ASSOC ", cleanname):
        cleanname += " PTA "
    if re.search(r" PTA ", cleanname):
        cleanname = cleanname.replace(" PARENTS ", " ")
        cleanname = cleanname.replace(" PARENT ", " ")
        cleanname = cleanname.replace(" TEACHERS ", " ")
        cleanname = cleanname.replace(" TEACHER ", " ")
        cleanname = cleanname.replace(" FRIENDS ", " ")
        cleanname = cleanname.replace(" FRIEND ", " ")
        cleanname = cleanname.replace(" STAFF ", " ")
        cleanname = cleanname.replace(" ASSOC ", " ")
    cleanname = cleanname.replace(" CA ", " COMMUNITY ASSOC ")
    return cleanname


def process_supplier_name(supplier_name):
    if supplier_name.startswith('"'):
        supplier_name = supplier_name[1:]
    if supplier_name.endswith('"'):
        supplier_name = supplier_name[:-1]
    supplier_name = re.sub(r'[^\x20-\x7E]', '', supplier_name)
    supplier_name = supplier_name.strip()
    cn = " " + supplier_name + " "
    clen = len(cn)

    tpos = 0
    while tpos <= clen - 5:
        if (cn[tpos] == " " and cn[tpos + 1] != " " and cn[tpos + 2] == " " and
            cn[tpos + 3] != " " and cn[tpos + 4] == " "):
            labbr = 4
            while (tpos + labbr + 1 < clen and cn[tpos + labbr + 1] != " " and
                   tpos + labbr + 2 < clen and cn[tpos + labbr + 2] == " "):
                labbr += 2
            abb_long = cn[tpos:tpos + labbr + 1]
            abb_short = " " + abb_long.replace(" ", "") + " "
            new_cn = cn.replace(abb_long, abb_short, 1)
            cn = new_cn
            clen = len(cn)
        tpos += 1
    supplier_name = cn.strip()
    return supplier_name


def normaliser(supplier_name):
    supplier_name = supplier_name.upper().replace('`S', "'S")
    for original, replacer in {"ST.": "ST ",
                               "ASSOC.": "ASSOC ",
                               "..": " ",
                               ".": "",
                               "(": " ",
                               ")": " ",
                               ",": " ",
                               ";": " ",
                               ":": " ",
                               "-": " ",
                               "/": " ",
                               "@": " ",
                               "+": " ",
                               "*": " ",
                               "[": " ",
                               "]": " ",
                               "!": " ",
                               "|": " ",
                               "`": " ",
                               "=": " ",
                               "\\": " ",
                               "_": " ",
                               "{": " ",
                               "}": " ",
                               "%": " ",
                               "&": " ",
                               "#": " "}.items():
        supplier_name = supplier_name.replace(original, replacer)
    suppliername = ' ' + supplier_name + ' '
    for original, replacer in {" 'S ": "S ",
                               "'S ": "S ",
                               "' ": " ",
                               " '": " ",
                               "S'": "S ",
                               "'N'": " N ",
                               "'": ""}.items():
        supplier_name = supplier_name.replace(original, replacer)
    supplier_name = re.sub(' +', ' ', supplier_name)
    for original, replacer in {"FREIND": "FRIEND",
                               " ASSOCATION": " ASSOCIATION",
                               " ASSOCIATON": " ASSOCIATION",
                               " ASOCIATION": " ASSOCIATION",
                               " ASSOCAITION": " ASSOCIATION",
                               " ASSOCIAION ": " ASSOCIATION ",
                               " ASSOCAITION": " ASSOCIATION",
                               " ASSOCIATOPM": " ASSOCIATION",
                               " ASS ": " ASSOCIATION ",
                               " ASSOCN ": " ASSOCIATION ",
                               " ASSOCIATIONSACDA ": " ASSOCIATION SACDA ",
                               " CENTER": " CENTRE",
                               "GIUDE": "GUIDE",
                               " CONGEGATION ": " CONGREGATION ",
                               " ORGANIZAT": " ORGANISAT",
                               " DISTICT ": " DISTRICT ",
                               " DISRICT ": " DISTRICT ",
                               " DISABILLI": " DISABILI",
                               " AMATUER ": " AMATEUR ",
                               " BUSISNESS ": " BUSINESS ",
                               " VICARIGE ": " VICARAGE ",
                               " REHABILITAION ": " REHABILITATION ",
                               " PANAL": " PANEL",
                               " BRITAN ": " BRITAIN ",
                               " BRITANIA ": " BRITANNIA ",
                               " CUMBRA ": " CUMBRIA ",
                               " NEIGHBOR": " NEIGHBOUR",
                               " COUNCILOR ": " COUNCILLOR ",
                               " MATHEW": " MATTHEW",
                               " VILLIAGE": " VILLAGE",
                               " HERATAGE ": " HERITAGE ",
                               " SHEILD": " SHIELD",
                               " COMUNITY ": " COMMUNITY ",
                               " COMUNITIES ": " COMMUNITIES ",
                               " COMMMUNITY ": " COMMUNITY ",
                               " COMITTEE": " COMMITTEE",
                               " COMMITEE": " COMMITTEE",
                               " INDEPENDANT ": " INDEPENDENT ",
                               " SYNDROMAE ": " SYNDROME ",
                               " WILDLIFW ": " WILDLIFE ",
                               " CENRE ": " CENTRE ",
                               " COMMUNITYYOUTH ": " COMMUNITY YOUTH ",
                               " AUTISMWEST ": " AUTISM WEST ",
                               " OFGOD ": " OF GOD ",
                               " INFORMARION ": " INFORMATION ",
                               " DEVELOPEMENT ": " DEVELOPMENT ",
                               " CHRITIAN ": " CHRISTIAN ",
                               " ROYALM ": " ROYAL ",
                               " LARYNGECOMY ": " LARYNGECTOMY ",
                               " ALCHOL ": " ALCOHOL ",
                               " RESARCH ": " RESEARCH ",
                               " REASEARCH ": " RESEARCH ",
                               " BEATY ": " BEAUTY ",
                               " CENTR ": " CENTRE "}.items():
        supplier_name = supplier_name.replace(original, replacer)
    supplier_name = re.sub(' +', ' ', supplier_name)

    for ltd in ['LIMITED', 'LIMITE', 'LIMIT', 'LIMIT ', 'LIMI', 'LIMI', 'LIM', 'LIM']:
        if supplier_name.strip().endswith(ltd):
            supplier_name = supplier_name.replace(ltd, 'LTD ')

    for original, replacer in {"PUBLIC LIMITED COMPANY": "PLC",
                               "C I C": "CIC",
                               "COMMUNITY INTEREST COMPANY": "CIC",
                               "COMMUNITY INTEREST COMPAN": "CIC",
                               "COMMUNITY INTEREST COMPA": "CIC",
                               "COMMUNITY INTEREST COMP": "CIC",
                               "COMMUNITY INTEREST COM": "CIC",
                               "COMMUNITY INTEREST CO": "CIC",
                               "COUNCIL FOR VOLUNTARY SERVICES": "CVS",
                               "COUNCIL FOR VOLUNTARY SERVICE": "CVS",
                               "UNITED REFORMED CHURCH": "URC",
                               "URC CHURCH": "URC",
                               "UR CHURCH": "URC",
                               "ALSO KNOWN AS": "AKA",
                               "ROYAL ANTEDILUVIAN ORDER OF BUFFALOES": "RAOB",
                               "ROYAL ANTIDILUVIAN ORDER OF BUFALLOES": "RAOB",
                               "ROYAL ANTEDILUVIAN ORDER OF BUFFALOS": "RAOB",
                               "CO OP": "COOPERATIVE",
                               "CO OPS": "COOPERATIVE",
                               "CO OPERATIVE": "COOPERATIVE",
                               "CO OPERATIVES": "COOPERATIVE",
                               "COOP ": "COOPERATIVE",
                               "COOPS": "COOPERATIVE",
                               "COOPERATIVES": "COOPERATIVE",
                               "DEPARTMENT": "DEPT",
                               "DEPARTMENTS": "DEPT",
                               "DEPTS": "DEPT",
                               "PROG ": "PROGRAMME",
                               "PROGRAM ": "PROGRAMME",
                               "ASSOCIATION": "ASSOC",
                               "COMM ": "COMMUNITY",
                               "SOCIETY": "SOC",
                               "SOCY": "SOC",
                               " SERV": " SERVICE",
                               "REGT": "REGIMENT",
                               " INFO": " INFORMATION",
                               " AVE": " AVENUE",
                               " THEATRE CO ": "THEATRE COMPANY",
                               "AND CO ": "AND COMPANY ",
                               "CO LTD": "COMPANY LTD"}.items():
        supplier_name = supplier_name.replace(original, replacer)

    if supplier_name.endswith(" CO "):
        temp_name = supplier_name + "#"
        temp_name = temp_name.replace(" CO #", " COMPANY #")
        supplier_name = temp_name.replace("#", "")

    for original, replacer in {" THE ": " ",
                               " AND ": " ",
                               " OF ": " ",
                               " FOR ": " ",
                               " WITH ": " ",
                               " AT ": " ",
                               " TO ": " ",
                               " IN ": " ",
                               " ON ": " ",
                               " AN ": " "}.items():
        supplier_name = supplier_name.replace(original, replacer)

    supplier_name = re.sub(' +', ' ', supplier_name)
    supplier_name = process_supplier_name(supplier_name)
    supplier_name = process_cleanname(supplier_name)
    supplier_name = re.sub(' +', ' ', supplier_name)

    for original, replacer in {"SCOUT GROUP": "SCOUTS",
                               " SCOUT ASSOC ": " SCOUTS ",
                               " SCOUTS ASSOC ": " SCOUTS ",
                               "SCOUT UNIT": "SCOUTS",
                               "SCOUT UNITS": "SCOUTS",
                               "SCOUTS UNIT": "SCOUTS",
                               "SCOUTS UNITS": "SCOUTS",
                               "SCOUT GROUP": "SCOUTS",
                               "SCOUT GROUPS": "SCOUTS",
                               "SCOUTS GROUP": "SCOUTS",
                               "SCOUTS GROUPS": "SCOUTS",
                               "SCOUT PACK": "SCOUTS",
                               "SCOUT PACKS": "SCOUTS",
                               "SCOUTS PACK": "SCOUTS",
                               "SCOUTS PACKS": "SCOUTS",
                               "BOY SCOUTS": "SCOUTS",
                               "GIRL GUIDE": "GIRL GUIDES",
                               "GIRL GUIDING": "GIRL GUIDES",
                               "GIRLGUIDING": "GIRL GUIDES",
                               "GIRL GUIDES": "GUIDES",
                               "GUIDE ASSOC": "GUIDES",
                               "GUIDES ASSOC": "GUIDES",
                               "GUIDE UNIT": "GUIDES",
                               "GUIDE UNITS": "GUIDES",
                               "GUIDES UNIT": "GUIDES",
                               "GUIDES UNITS": "GUIDES",
                               "GUIDE GROUP": "GUIDES",
                               "GUIDE GROUPS": "GUIDES",
                               "GUIDES GROUP": "GUIDES",
                               "GUIDES GROUPS": "GUIDES",
                               "GUIDE PACK": "GUIDES",
                               "GUIDE PACKS": "GUIDES",
                               "GUIDES PACK": "GUIDES",
                               "GUIDES PACKS": "GUIDES",
                               "BROWNIE ASSOC ": "BROWNIES ",
                               "BROWNIES ASSOC ": "BROWNIES ",
                               "BROWNIE UNIT": "BROWNIES",
                               "BROWNIE UNITS": "BROWNIES",
                               "BROWNIES UNIT": "BROWNIES",
                               "BROWNIES UNITS": "BROWNIES",
                               "BROWNIE GROUP": "BROWNIES",
                               "BROWNIE GROUPS": "BROWNIES",
                               "BROWNIES GROUP": "BROWNIES",
                               "BROWNIES GROUPS": "BROWNIES",
                               "BROWNIE PACK": "BROWNIES",
                               "BROWNIE PACKS": "BROWNIES",
                               "BROWNIES PACK": "BROWNIES",
                               "BROWNIES PACKS": "BROWNIES",
                               "BEAVER GROUP": "BEAVERS",
                               "BEAVER GROUPS": "BEAVERS",
                               "BEAVERS GROUP": "BEAVERS",
                               "BEAVERS GROUPS": "BEAVERS",
                               "BEAVER COLONY": "BEAVERS",
                               "BEAVERS COLONY": "BEAVERS",
                               "CESCHOOL": "CE SCHOOL",
                               "CPSCHOOL": "CP SCHOOL",
                               "RCSCHOOL": "RC SCHOOL",
                               "PRE SCHOOL": "PRESCHOOL",
                               "PLAY SCHOOL": "PLAYSCHOOL",
                               "WOMENS INSTITUTE": "WI",
                               "WOMEN INSTITUTE": "WI",
                               "WORKINGMENS": "WORKING MENS",
                               "WORKING MENS SOCIAL CLUB": "WMC",
                               "WORKING MENS CLUB": "WMC",
                               "WORKMENS CLUB": "WMC",
                               "WMC INSTITUTE": "WMC",
                               "SAINT": "ST",
                               "NORTH EAST": "NE",
                               "NORTH WEST": "NW",
                               "SOUTH EAST": "SE",
                               "SOUTH WEST": "SW",
                               "NORTHEAST": "NE",
                               "NORTHWEST": "NW",
                               "SOUTHEAST": "SE",
                               "SOUTHWEST": "SW",
                               "STH": "SOUTH",
                               "COF E": "CE",
                               "C OFE": "CE",
                               "COFE": "CE",
                               "MIDDLESEX": "MIDDX",
                               "BEDFORDSHIRE": "BEDS",
                               "BERKSHIRE": "BERKS",
                               "BUCKINGHAMSHIRE": "BUCKS",
                               "CAMBRIDGESHIRE": "CAMBS",
                               "HUNTINGDONSHIRE": "HUNTS",
                               "CHESHIRE": "CHES",
                               "DERBYSHIRE": "DERBYS",
                               "CO DURHAM": "COUNTY DURHAM",
                               "GLOUCESTERSHIRE": "GLOS",
                               "HAMPSHIRE": "HANTS",
                               "HAMPS": "HANTS",
                               "HEREFORDSHIRE": "HEREFS",
                               "HERTFORDSHIRE": "HERTS",
                               "ISLE OF WIGHT": "IOW",
                               "ISLE WIGHT": "IOW",
                               "LANCASHIRE": "LANCS",
                               "LEICESTERSHIRE": "LEICS",
                               "LINCOLNSHIRE": "LINCS",
                               "NORTHAMPTONSHIRE": "NORTHANTS",
                               "NLAND": "NORTHUMBERLAND",
                               "NOTTINGHAMSHIRE": "NOTTS",
                               "OXFORDSHIRE": "OXON",
                               "SHROPSHIRE": "SALOP",
                               "SHROPS": "SALOP",
                               "STAFFORDSHIRE": "STAFFS",
                               "WARWICKSHIRE": "WARKS",
                               "WILTSHIRE": "WILTS",
                               "WORCESTERSHIRE": "WORCS",
                               "YORKSHIRE": "YORKS",
                               "SOUTHAMPTION": "SOUTHAMPTON",
                               "SOTHAMPTON": "SOUTHAMPTON",
                               "BRIMINGHAM": "BIRMINGHAM",
                               "BHAM": "BIRMINGHAM",
                               "BIRMINGAHM": "BIRMINGHAM",
                               "PERY BARR": "PERRY BARR",
                               "GLAGOW": "GLASGOW",
                               "FOOTBALL CLUB": "FC",
                               "YOUNG MENS CHRISTIAN ASSOCIATION": "YMCA",
                               "YOUNG WOMENS CHRISTIAN ASSOCIATION": "YWCA",
                               "YOUNG MENS CHRISTIAN ASSOC": "YMCA",
                               "YOUNG WOMENS CHRISTIAN ASSOC": "YWCA",
                               "INCORPORATED": "INC",
                               " AFC": " FC",
                               "JUNIORS FC": "JUNIOR FC",
                               "JFC": "JUNIOR FC",
                               "ARLFC": "AMATEUR RUGBY LEAGUE FC",
                               "RUFC": "RUGBY UNION FC",
                               "RLFC": "RUGBY LEAGUE FC",
                               "RFC": "RUGBY FC",
                               "YFC": "YOUTH FC",
                               "NEWCASTLE UPON TYNE": "NEWCASTLE",
                               "NEWCASTLE TYNE": "NEWCASTLE",
                               "HOLME UPON SPALDING MOOR": "HOLME SPALDING MOOR",
                               "UPON": "",
                               "1ST": "FIRST",
                               "IST": "FIRST",
                               "2ND": "SECOND",
                               "3RD": "THIRD",
                               "FOURTH": "4",
                               "FIFTH": "5",
                               "SIXTH": "6",
                               "SEVENTH": "7",
                               "EIGHTH": "8",
                               "NINTH": "9",
                               "TENTH": "10",
                               "ELEVENTH": "11",
                               "TWELFTH": "12",
                               "THIRTEENTH": "13",
                               "FOURTEENTH": "14",
                               "FIFTEENTH": "15",
                               "SIXTEENTH": "16",
                               "SEVENTEENTH": "17",
                               "EIGHTEENTH": "18",
                               "NINETEENTH": "19",
                               "TWENTIETH": "20"}.items():
        supplier_name = supplier_name.replace(original, replacer)
    if 'GUIDE' in supplier_name:
        supplier_name = supplier_name.replace(" BROWNIE "," BROWNIES ")
        supplier_name = supplier_name.replace(" SCOUT "," SCOUTS ")
    if 'SCOUT' in supplier_name:
        supplier_name = supplier_name.replace(" CUB "," CUBS ")
        supplier_name = supplier_name.replace(" SCOUT SCOUT "," SCOUT ")
    if 'SCHOOL' in supplier_name:
        supplier_name = supplier_name.replace(" ROMAN CATHOLIC "," RC ")
        supplier_name = supplier_name.replace(" CATHOLIC "," RC ")
        supplier_name = supplier_name.replace(" CHURCH ENGLAND "," CE ")
        supplier_name = supplier_name.replace(" JUNIOR INFANT "," JI ")


    if re.search(r" FIRST ", supplier_name):
        supplier_name += " 1 "
    if re.search(r" SECOND ", supplier_name):
        supplier_name += " 2 "
    if re.search(r" THIRD ", supplier_name):
        supplier_name += " 3 "

    for tdigit in range(10):
        ststr = f"{tdigit}ST "
        ndstr = f"{tdigit}ND "
        rdstr = f"{tdigit}RD "
        thstr = f"{tdigit}TH "
        newstr = f"{tdigit} "

        supplier_name = supplier_name.replace(ststr, newstr)
        supplier_name = supplier_name.replace(ndstr, newstr)
        supplier_name = supplier_name.replace(rdstr, newstr)
        supplier_name = supplier_name.replace(thstr, newstr)

    supplier_name = supplier_name.replace(" CO OPERAT"," COOPERAT")
    supplier_name = supplier_name.replace(" CO ORDINAT"," COORDINAT")
    supplier_name = supplier_name.replace(" PRIMARY CARE TRUST"," PCT")
    supplier_name = supplier_name.replace(" NATIONAL HEALTH SERVICE "," NHS ")
    supplier_name = supplier_name.replace(" A "," ")
    supplier_name = re.sub(' +', ' ', supplier_name)
    supplier_name = supplier_name.replace('"', '')
    supplier_name = supplier_name.strip()
    return supplier_name.upper()
