"""
从数据库中查询数据
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
from scipy import stats
# import statsmodels.api as smapi

# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# 根据本地的Bond_convValue_csv和Bond_pureValue_csv文件计算可转债的平价溢价率
def calculate_PSPremiumRate_local(config, BondInfo, mark):
    print('calculate_PSPremiumRate_local...')
    #
    df = pd.read_csv(config['Info_dir'] + 'Bond_convValue_csv.csv')
    mapping_dict = dict(zip(BondInfo['SYMBOL'], BondInfo['SYMBOL9']))
    cols_new = [mapping_dict.get(item, item) for item in df.columns]
    df.columns = cols_new
    df = pd.melt(df, id_vars=['date'], var_name='SYMBOL9', value_name='convValue')
    df['date'] = pd.to_datetime(df['date'])
    df = pd.merge(mark, df, on=['date', 'SYMBOL9'], how='left')
    # df2 = df[df['convValue'].isna()]
    df['convValue'] = df.groupby(['SYMBOL9'])['convValue'].fillna(method='ffill')
    df = df.dropna(subset=['convValue'])
    df = df.drop(['flag'], axis=1)
    df_cv = df
    #
    # df = pd.read_csv('G:/Matlab_Documents/Matlab_STrade/STK_inDATA/CST_STKAll_Info/Bond_pureValue_csv.csv')
    df = pd.read_csv(config['Info_dir'] + 'Bond_pureValue_csv.csv')
    mapping_dict = dict(zip(BondInfo['SYMBOL'], BondInfo['SYMBOL9']))
    cols_new = [mapping_dict.get(item, item) for item in df.columns]
    if cols_new[0] == 'DATE':
        cols_new[0] = 'date'
    df.columns = cols_new
    df = pd.melt(df, id_vars=['date'], var_name='SYMBOL9', value_name='pureValue')
    df['date'] = pd.to_datetime(df['date'])
    df = pd.merge(mark, df, on=['date', 'SYMBOL9'], how='left')
    df['pureValue'] = df.groupby(['SYMBOL9'])['pureValue'].fillna(method='ffill')
    df = df.drop(['flag'], axis=1)
    df = df.dropna(subset=['pureValue'])
    df_pv = df
    #
    df = pd.merge(df_cv, df_pv, on=['date', 'SYMBOL9'])
    df['PSPremiumRate'] = df['convValue'] / df['pureValue'] - 1

    return df

# 按天滚动计算col1和col2列的秩相关系数
def calculate_rank_correlation(df, col1, col2):
    print('calculate_rank_correlation...')
    df.set_index('date', inplace=True)  # 将日期列设为索引
    def correlation(group):
        # 计算秩相关系数
        rho, p_value = stats.spearmanr(group[col1], group[col2])
        # return pd.Series({'rho': rho, 'p_value': p_value})
        return pd.Series({'RankIC': rho})

    result = df.groupby(level=0).apply(correlation)  # 按天分组计算相关系数
    return result

def calculate_rank_ic(df):
    # df['corr'] = df.groupby(pd.Grouper(key='TRADINGDATE', freq='D'))['CHANGERATIO', 'FACTORVALUE_LastDay'].corr()
    # df3 = df.groupby('TRADINGDATE')['CHANGERATIO', 'FACTORVALUE_LastDay'].corr()
    df_out = pd.DataFrame(df['TRADINGDATE'].unique())
    df_out.columns = ['TRADINGDATE']
    df_out['RankIC'] = np.nan
    for nD in range(0, len(df_out['TRADINGDATE'])):
        date = df_out['TRADINGDATE'].iloc[nD]
        df3 = df.loc[df['TRADINGDATE'] == date]
        spearman_corr, _ = stats.spearmanr(df3['FACTORVALUE_LastDay'], df3['CHANGERATIO'])
        df_out['RankIC'].iloc[nD] = spearman_corr

    return df_out


def convert_date_format_yyyymmdd2Y_M_D(date_str):
    # 假设输入的日期字符串是8位yyyymmdd格式
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    date_obj = datetime(year, month, day)
    return date_obj.strftime('%Y-%m-%d')


# 从字符串因子名中删除扩展字段
def delete_opt_from_factorname(factorname, opt):
    opt3 = []
    factorname2 = factorname
    for opt2 in opt:
        if opt2 in factorname:
            factorname2 = factorname.replace('_' + opt2, '')
            opt3 = opt2
    return factorname2, opt3


# 从Info文件中剔除掉非目标股池代码中的个股数据
def dropSTK_not_inTargetPool(df, STKlist):
    symbol = df['SYMBOL'].unique()
    count = 0
    for code in symbol:
        count = count + 1
        if count % 500 == 1:
            print(code)
        if code in STKlist:
            a = 1
        else:
            # print(code)
            # rows = rows[rows['SYMBOL'] != code]
            df = df.drop(df[df['SYMBOL'] == code].index)
    df = df.reset_index(drop=True)

    # df = df[df[SYMBOL].isin(STKlist)]

    return df


# MAD法去极值
def extremum(df, cols, madvalue):
    def group_operation(group):
        median = np.median(group)
        mad = np.median(np.abs(group - median))
        upper_bound = median + madvalue * mad
        lower_bound = median - madvalue * mad
        return group.clip(lower=lower_bound, upper=upper_bound)

    # 按列 A 进行分组，并对列 B 应用 MAD 法去极值
    for col in cols:
        df[col + '_ext'] = df.groupby('date')[col].transform(group_operation)

    return df


# 提取日历表中的月度标签
def fetchCalendard2Mth(TradeDays):
    TradeDays = TradeDays.drop_duplicates(subset='CALENDARDATE', keep='first')
    TradeDays = TradeDays.reset_index(drop=True)
    TradeDays['TradeMth'] = TradeDays['CALENDARDATE'].apply(lambda x: int(x.strftime('%Y%m')))

    return TradeDays

#
def fillByGroup(df, groupcol, datecol, opt):
    # groupcol用来分组的列 datecol需要填充的列
    for col in datecol:
        df[col] = df.groupby(groupcol)[col].fillna(method='ffill')

    return df


# 计算可转债池的风格标记，每天
def getCBPoolStyle(config, BondInfo, mark):
    print(config['pool_mark'], 'getCBPoolStyle...')
    #
    if config['pool_mark'] == 'WM':
        df = mark
        df[config['pool_mark']] = 1
    if config['pool_mark'] in ['GP', 'ZP', 'DP']:
        df = calculate_PSPremiumRate_local(config, BondInfo, mark)
        df['date'] = pd.to_datetime(df['date'])
        df = getCBPoolStyle_GZDP(df)
        df = df[['date', 'SYMBOL9', config['pool_mark'] + '_LD']]
        df.rename(columns={config['pool_mark'] + '_LD': config['pool_mark']}, inplace=True)
    if config['pool_mark'] in ['GQ', 'ZQ', 'DQ']:
        df = getCBPoolStyle_GZDQ(mark)
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'SYMBOL9', config['pool_mark'] + '_LD']]
        df.rename(columns={config['pool_mark'] + '_LD': config['pool_mark']}, inplace=True)
    if config['pool_mark'] in ['GQ50', 'GQ80', 'ZQm', 'DQ150']:
        df = getCBPoolStyle_GZDQ_50_80_150(mark)
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'SYMBOL9', config['pool_mark'] + '_LD']]
        df.rename(columns={config['pool_mark'] + '_LD': config['pool_mark']}, inplace=True)
    df = df[['date', 'SYMBOL9', config['pool_mark']]].copy()
    df.rename(columns={config['pool_mark']: 'PoolMark'}, inplace=True)
    df = df[(df['date'] >= config['start_time']) & (df['date'] <= config['end_time'])].reset_index(drop=True)

    return df

#
def getCBPoolStyle_GZDP(df):
    print('getCBPoolStyle_GZDP...')
    # 定义一个函数，按1:2:1的比例分组
    def split_into_three_groups(group):
        group = group.sort_values(by='PSPremiumRate').reset_index(drop=True)  # 按列B排序
        n = len(group)
        # 计算分组的索引
        idx1 = n // 4  # 第一组结束位置
        idx2 = idx1 + n // 2  # 第二组结束位置
        # 分配组标签
        group['group_label'] = np.where(
            group.index <= group.index[idx1], 'Group1',
            np.where(group.index <= group.index[idx2], 'Group2', 'Group3')
        )
        return group
    # 按列A分组并应用分组函数
    result = df.groupby('date').apply(split_into_three_groups).reset_index(drop=True)
    # df2 = result[result['date'] == '2024-07-01']
    result[['DP', 'ZP', 'GP']] = np.nan
    result.loc[result['group_label'] == 'Group1', 'DP'] = 1
    result.loc[result['group_label'] == 'Group2', 'ZP'] = 1
    result.loc[result['group_label'] == 'Group3', 'GP'] = 1
    result = result.drop(['group_label'], axis=1)
    result['DP_LD'] = result.groupby('SYMBOL9')['DP'].shift(1)
    result['ZP_LD'] = result.groupby('SYMBOL9')['ZP'].shift(1)
    result['GP_LD'] = result.groupby('SYMBOL9')['GP'].shift(1)

    return result

def getCBPoolStyle_GZDQ(df):
    print('getCBPoolStyle_GZDQ...')
    # 定义一个函数，按1:2:4的比例分组
    def group_with_ratio(sub_df):
        sub_df = sub_df.sort_values(by='MKV', ascending=False)
        total_length = len(sub_df)
        first_group_size = int(total_length * 1 / (1 + 2 + 4))
        second_group_size = int(total_length * 2 / (1 + 2 + 4))
        labels = []
        for i in range(total_length):
            if i < first_group_size:
                labels.append('Group1')
            elif i < first_group_size + second_group_size:
                labels.append('Group2')
            else:
                labels.append('Group3')
        sub_df['group_label'] = labels
        return sub_df
    # 按天分组并应用分组函数
    result = df.groupby(df['date']).apply(group_with_ratio).reset_index(drop=True)
    # df2 = result[result['date'] == '2024-07-01']
    result[['GQ', 'ZQ', 'DQ']] = np.nan
    result.loc[result['group_label'] == 'Group1', 'GQ'] = 1
    result.loc[result['group_label'] == 'Group2', 'ZQ'] = 1
    result.loc[result['group_label'] == 'Group3', 'DQ'] = 1
    result = result.drop(['group_label'], axis=1)
    result['GQ_LD'] = result.groupby('SYMBOL9')['GQ'].shift(1)
    result['ZQ_LD'] = result.groupby('SYMBOL9')['ZQ'].shift(1)
    result['DQ_LD'] = result.groupby('SYMBOL9')['DQ'].shift(1)

    return result


def getCBPoolStyle_GZDQ_50_80_150(df):
    print('getCBPoolStyle_GZDQ...')
    df[['GQ50', 'GQ80', 'ZQm', 'DQ150']] = np.nan
    # 定义一个函数，按指定个数分组
    def group_with_ratio(sub_df):
        sub_df = sub_df.sort_values(by='MKV', ascending=False)
        sub_df = sub_df.reset_index(drop=True)
        total_length = len(sub_df)
        sub_df.loc[:49, 'GQ50'] = 1
        sub_df.loc[:79, 'GQ80'] = 1
        sub_df.loc[total_length - 150:, 'DQ150'] = 1
        #
        sub_df.loc[:, 'ZQm'] = 1
        sub_df.loc[:79, 'ZQm'] = np.nan
        sub_df.loc[total_length - 150:, 'ZQm'] = np.nan

        return sub_df
    # 按天分组并应用分组函数
    result = df.groupby(df['date']).apply(group_with_ratio).reset_index(drop=True)
    # df2 = result[result['date'] == '2024-07-01']
    # sub_df = df[df['date'] == '2024-07-01']
    # sub_df = sub_df.fillna(0)
    # te=sub_df['GQ80']+sub_df['ZQm']+sub_df['DQ150']

    result['GQ50_LD'] = result.groupby('SYMBOL9')['GQ50'].shift(1)
    result['GQ80_LD'] = result.groupby('SYMBOL9')['GQ80'].shift(1)
    result['ZQm_LD'] = result.groupby('SYMBOL9')['ZQm'].shift(1)
    result['DQ150_LD'] = result.groupby('SYMBOL9')['DQ150'].shift(1)

    return result


# 将yaml文件中字典格式的因子列表转成通表
def getFactorList_group2list(FactorList):
    #
    FactorList2 = list(FactorList)
    FactorList3 = []
    for nFL in range(0, len(FactorList)):
        factorname = FactorList2[nFL]
        FactorList3 = FactorList3 + FactorList[factorname]
    #
    FactorList4 = []
    for temp in FactorList3:
        if isinstance(temp, list):
            FactorList4 = FactorList4 + temp
        elif isinstance(temp, str):
            FactorList4 = FactorList4 + [temp]
    FactorList3 = FactorList4
    # 剔除有生成，但是不需要检验的因子
    list_dlt = ['Fund_NewBnd', 'Fund_FlagST']
    FactorList3 = [x for x in FactorList3 if x not in list_dlt]
    FactorList3.sort()

    return FactorList3


# 将yaml文件中字典格式的因子列表转成通表
def getFactorList_group2list_v2(FactorList):
    #
    FactorList2 = list(FactorList)
    FactorList3 = []
    for nFL in range(0, len(FactorList)):
        factorname = FactorList2[nFL]
        FactorList3 = FactorList3 + FactorList[factorname]
    # 剔除有生成，但是不需要检验的因子
    list_dlt = ['Fund_NewBnd', 'Fund_FlagST']
    FactorList3 = [x for x in FactorList3 if x not in list_dlt]
    FactorList3.sort()

    return FactorList3


# 将list格式的因子列表进行字段扩展，增加Abs、Rcp、Price、MKV、TTM等字段
def getFactorList_extendlist(FactorList3, newlist):
    # newlist = ['MKV', 'Price', 'Rcp', 'Abs', 'MKV_Rcp', 'MKV_Abs', 'Price_Rcp', 'Price_Abs']
    # newlist = ['MKV', 'Price', 'Rcp', 'Abs', 'MKV_Rcp', 'MKV_Abs', 'Price_Rcp', 'Price_Abs', 'TTMd', 'TTMr', 'LYRd', 'LYRr', 'Neut']
    FactorList4 = FactorList3
    for newstr in newlist:
        newlist2 = [item + '_' + newstr for item in FactorList3]
        FactorList4 = FactorList4 + newlist2
    FactorList4.sort()

    return FactorList4

# 将yaml文件中字典格式的因子列表转成DataFrame格式的通表，并增加处理字段的因子名，和每个因子对应的原始因子
def getFactorList_group2pandas(FactorList, newlist):
    FactorList2 = list(FactorList)
    FactorList3 = []
    for nFL in range(0, len(FactorList)):
        factorname = FactorList2[nFL]
        FactorList3 = FactorList3 + FactorList[factorname]
    FactorList4 = []
    for temp in FactorList3:
        if isinstance(temp, list):
            FactorList4 = FactorList4 + temp
        elif isinstance(temp, str):
            FactorList4 = FactorList4 + [temp]
    FactorList3 = FactorList4
    # 剔除有生成，但是不需要检验的因子
    list_dlt = ['Fund_NewBnd', 'Fund_FlagST']
    FactorList3 = [x for x in FactorList3 if x not in list_dlt]
    FactorList3.sort()
    #
    FactorList2 = FactorList3
    FactorList3 = pd.DataFrame(FactorList2, columns={'FactorName'})
    FactorList3['ClassName'] = FactorList2
    #
    # newlist = ['MKV', 'Price', 'Rcp', 'Abs', 'MKV_Rcp', 'MKV_Abs', 'Price_Rcp', 'Price_Abs']
    # newlist = ['MKV', 'Price', 'Rcp', 'Abs', 'MKV_Rcp', 'MKV_Abs', 'Price_Rcp', 'Price_Abs', 'TTMd', 'TTMr', 'LYRd', 'LYRr', 'Neut', 'TSrank20', 'TSrank60']
    # newlist = config['factor_opt']
    if len(newlist) > 0:
        for newstr in newlist:
            # print(newstr)
            newlist2 = [item + '_' + newstr for item in FactorList2]
            # print(newlist2)
            newlist3 = pd.DataFrame(list(zip(newlist2, FactorList2)), columns=['FactorName', 'ClassName'])
            FactorList3 = pd.concat([FactorList3, newlist3], ignore_index=True)
    FactorList3.sort_values('FactorName', inplace=True)  # inplace=True表示对原始DataFrame进行排序
    FactorList3 = FactorList3.reset_index(drop=True)

    return FactorList3


# 读取本地的BondInfo表
def load_BondInfo(config):
    BondInfo = pd.read_csv(config['Info_dir'] + 'BOND_ConvertInfo.csv', dtype=str)
    BondInfo = BondInfo.drop(['SECURITYID', 'ENFULLNAME', 'ISSUERFULLNAME', 'PRICEADJUSTTERMS', 'PRICEDOWNWARDADJUSTTERMS', 'CALLTERMS', 'CALLCONDITION'], axis=1)
    BondInfo = BondInfo.drop(['FULLNAME', 'ISSUETYPE', 'CALLPRICE', 'PUTTERMS', 'PUTPRICE', 'ADDITIONALPUTTERMS', 'COMPENSATEINTERESTTERMS'], axis=1)
    BondInfo = BondInfo.drop(['PUTCONDITION', 'CREDITSTATUS', 'BONDSECURITYSTEP', 'CALLMATURITY'], axis=1)
    #
    indinfo = pd.read_csv(config['Info_dir'] + 'STK_INDUSTRYCLASS.csv', dtype=str)
    indinfo = indinfo[['SYMBOL9', 'INDUSTRYCODE1', 'INDUSTRYNAME1']]
    indinfo.rename(columns={'SYMBOL9': 'OBJECTSTOCKCODE9', 'INDUSTRYCODE1': 'INDUSTRYCODE', 'INDUSTRYNAME1': 'INDUSTRYNAME'}, inplace=True)
    BondInfo = pd.merge(BondInfo, indinfo, on=['OBJECTSTOCKCODE9'])

    return BondInfo


def load_CBfactor(config, BondInfo, TradeDays, factorname):
    newlist = ['MKV_Rcp', 'MKV_Abs', 'Price_Rcp', 'Price_Abs', 'Neut_MI', 'Neut_MP']
    dir2 = []
    for x in newlist:
        if x in factorname and factorname.endswith(x):
            dir2 = x
    if len(dir2) == 0:
        # newlist = ['MKV', 'Price', 'Rcp', 'Abs', 'Neut']
        newlist = ['MKV', 'Price', 'Rcp', 'Abs', 'TSrank120', 'TSrank60', 'TSzs120']
        for x in newlist:
            if x in factorname and factorname.endswith(x):
                dir2 = x
    if len(dir2) == 0:
        file_path = f"{config['factor_dir']}Array/{factorname}.csv"
    else:
        factorname = factorname[:-len('_' + dir2)]
        file_path = f"{config['factor_dir']}Array/{dir2}/{factorname}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'SYMBOL9' in df.columns and 'SYMBOL' not in df.columns:
            df.rename(columns={'SYMBOL9': 'SYMBOL'}, inplace=True)
    else:
        print('文件不存在,', file_path)
        df = []
        return df
    df['FACTORVALUE_LastDay'] = df.groupby('SYMBOL')['FACTORVALUE'].shift(1)
    df = df[(df['TRADINGDATE'] >= config['start_time']) & (df['TRADINGDATE'] <= config['end_time'])]

    STKlist = BondInfo['SYMBOL9'].tolist()
    df = df[df['SYMBOL'].isin(STKlist)]

    return df

def load_STKfactor(config, BondInfo, TradeDays, factorname):
    print('load_STKfactor', factorname)
    newlist = ['MKV_Rcp', 'MKV_Abs', 'Price_Rcp', 'Price_Abs']
    dir2 = []
    for x in newlist:
        if x in factorname and factorname.endswith(x):
            dir2 = x
    if len(dir2) == 0:
        # newlist = ['MKV', 'Price', 'Rcp', 'Abs']
        newlist = ['MKV', 'Price', 'Rcp', 'Abs', 'TTMd', 'TTMr', 'LYRd', 'LYRr', 'Neut']
        for x in newlist:
            if x in factorname and factorname.endswith(x):
                dir2 = x
    if len(dir2) == 0:
        file_path = f"{config['factor_dir']}Array/{factorname}.csv"
    else:
        factorname = factorname[:-len('_' + dir2)]
        file_path = f"{config['factor_dir']}Array/{dir2}/{factorname}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = []
        return df
    cols = df.columns
    if 'SYMBOL9' in cols and 'SYMBOL' not in cols:
        df.rename(columns={'SYMBOL9': 'SYMBOL'}, inplace=True)
    df['FACTORVALUE_LastDay'] = df.groupby('SYMBOL')['FACTORVALUE'].shift(1)
    df = df[(df['TRADINGDATE'] >= config['start_time']) & (df['TRADINGDATE'] <= config['end_time'])]

    STKlist = BondInfo['OBJECTSTOCKCODE9'].tolist()
    # if len(dir2) == 0:
    #     STKlist = BondInfo['OBJECTSTOCKCODE9'].tolist()
    # else:
    #     STKlist = BondInfo['SYMBOL9'].tolist()
    df = df[df['SYMBOL'].isin(STKlist)]
    df2 = []
    for nS in range(0, len(BondInfo)):
        codeCB = BondInfo['SYMBOL9'][nS]
        codeSTK = BondInfo['OBJECTSTOCKCODE9'][nS]
        # df3 = df[df['SYMBOL'] == codeSTK]
        df3 = df.loc[df['SYMBOL'] == codeSTK]
        if len(df3) > 0:
            # df3['SYMBOL'] = codeCB
            df3.loc[:, 'SYMBOL'] = codeCB
        if len(df2) == 0:
            df2 = df3
        else:
            df2 = pd.concat([df2, df3], ignore_index=True)

    return df2

# 读取本地的可转债日线信息表
def load_CBInfo(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['CLOSEPRICE'])
    df = df[df['AMOUNT'] > 0]
    return df

def load_CBInfo_all(config, BondInfo, cols, TradeDays):
    print('load_CBInfo_all...')
    # 读取因子检验时所需的个债行情
    df_all = []
    for nS in range(0, len(BondInfo)):
        codeCB = BondInfo['SYMBOL9'][nS]
        codeSTK = BondInfo['OBJECTSTOCKCODE9'][nS]
        if nS%500 == 1:
            print('load_CBInfo_all', nS, codeCB)
        #
        file_path = os.path.join(config['Info_dir'], 'Bond/', codeCB + '.csv')
        df = []
        if os.path.exists(file_path):
            # print(file_path)
            df = load_CBInfo(file_path)
            df['SYMBOL9_STK'] = codeSTK
            if 'SYMBOL9' in df.columns and 'SYMBOL' not in df.columns:
                df.rename(columns={'SYMBOL9': 'SYMBOL'}, inplace=True)
            # 填充最新市值
            tempNum = BondInfo[BondInfo['SYMBOL9'] == codeCB]['ISSUEQUANTITY'].values.astype(float) * 100000000 / 100
            df['ISSUEQUANTITY'] = tempNum[0]
            df['MKV'] = df['CLOSEPRICE'] * tempNum
            #
            df = df[(df['TRADINGDATE'] >= config['start_time']) & (df['TRADINGDATE'] <= config['end_time'])]
        if len(df) == 0:
            # print(config['Info_dir'], 'Bond数据为空，', codeCB)
            continue
        if len(df_all) == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)

    df_all['SYMBOL9'] = df_all['SYMBOL'].astype(str) + '.' + df_all['EXCHANGECODE']
    # df_all = df_all[['SYMBOL9', 'TRADINGDATE', 'CLOSEPRICE', 'AMOUNT', 'CHANGERATIO']]
    df_all.rename(columns={'CHANGERATIO': 'return'}, inplace=True)
    df_all = df_all[cols]

    return df_all

# 读取本地的正股日线信息表
def load_STKInfo(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['CLOSEPRICE'])
    df = df[df['AMOUNT'] > 0]
    return df

def load_STKInfo_all(config, BondInfo, TradeDays):
    # 读取因子生成时所需的正股行情
    df_all = []
    for nS in range(0, len(BondInfo)):
        codeCB = BondInfo['SYMBOL9'][nS]
        codeSTK = BondInfo['OBJECTSTOCKCODE9'][nS]
        if nS%100 == 1:
            print('load_STKInfo_all', codeSTK)
        #
        file_path = os.path.join(config['Info_dir'], 'STK_Org/', codeSTK + '.csv')
        # 读取数据
        df = []
        if os.path.exists(file_path):
            # print(file_path)
            df = load_STKInfo(file_path)
            df = df[(df['TRADINGDATE'] >= config['start_time']) & (df['TRADINGDATE'] <= config['end_time'])]
            # df['SYMBOL9_STK'] = df['SYMBOL9']
            df['SYMBOL9'] = codeCB
        if len(df) == 0:
            print(config['Info_dir'], 'STK数据为空，', codeSTK)
            continue
        if len(df_all) == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)

    df_all = df_all[['SYMBOL9', 'TRADINGDATE', 'CLOSEPRICE', 'AMOUNT', 'CHANGERATIO', 'MARKETVALUE']]
    df_all.rename(columns={'CHANGERATIO': 'return_STK', 'MARKETVALUE': 'MKV_STK', 'CLOSEPRICE': 'CLOSEPRICE_STK', 'AMOUNT': 'AMOUNT_STK'}, inplace=True)

    return df_all

def load_tradeday(config):
    df = pd.read_csv(config['Info_dir'] + 'STK_CALENDARD.csv', dtype=str, parse_dates=['CALENDARDATE'])
    df = df[(df['CALENDARDATE'] >= config['start_time']) & (df['CALENDARDATE'] <= config['end_time'])]
    df = df[df['ISOPEN'] == 'Y']
    df = df[(df['EXCHANGECODE'] == 'SSE') | (df['EXCHANGECODE'] == 'SZSE')]
    return df


def multiCycle_rolling_dev(df, col_data, outname, windows, optv):
    # df.rolling滚动取mean/std
    for cycle in windows:
        new = outname + str(int(cycle))
        #
        # df[new] = df[col_data].rolling(window=cycle, min_periods=1).mean() / df[col_data].rolling(window=cycle, min_periods=1).std()
        #
        df[new] = df[col_data].rolling(window=cycle, min_periods=1).mean()
        if optv == 0:
            df[new] = (df[col_data] - df[new]) / df[new]
        else:
            df[new] = (df[col_data] - df[new]) / (df[new] + optv)
        #
        df[new] = df[new].replace([np.inf, -np.inf, np.nan], 0)

    return df

def multiCycle_rolling_skew(df, col_data, outname, windows):
    # df.rolling滚动取skew
    # 用的是pd自带的skew函数
    for cycle in windows:
        df[outname + str(cycle)] = df[col_data].rolling(window=cycle, min_periods=1).skew()
    return df

def multiCycle_rolling_skew_for(df, col_data, outname, windows):
    # 手动for循环按滚动周期计算skew
    # 用的是scipy.stats带的skew函数
    df2 = df[col_data]
    for cycle in windows:
        df[outname + str(cycle)] = np.nan
        if len(df) < cycle:
            continue
        rolling_skew = [stats.skew(df2[i:i + cycle]) for i in range(len(df2) - cycle + 1)]
        df[outname + str(cycle)][cycle - 1:] = rolling_skew
    return df

def multiCycle_rolling_std(df, col_data, outname, windows):
    # df.rolling滚动取sum
    for cycle in windows:
        df[outname + str(cycle)] = df[col_data].rolling(window=cycle, min_periods=1).std()
    return df

def multiCycle_rolling_sum(df, col_data, outname, windows):
    # df.rolling滚动取sum
    for cycle in windows:
        df[outname + str(cycle)] = df[col_data].rolling(window=cycle, min_periods=1).sum()
        # df[outname + str(cycle)] = df[col_data].rolling(window=cycle, min_periods=1).sum()
    return df

def multiCycle_shift_div(df, col_data, outname, windows):
    # 滚动取变动幅度
    for cycle in windows:
        df[outname + str(cycle)] = df[col_data] / df[col_data].shift(cycle) - 1
    return df


# 对因子进行风格中性化
def neutralization(df, BondInfo, col, opt):
    # BondInfo用来提供行业信息
    if opt == 'Neut_IND' or opt == 'Neut_MI':
        industry = BondInfo['INDUSTRYCODE'].unique().tolist()
        df[industry] = 0
        for nC in range(0, len(BondInfo)):
            code = BondInfo['OBJECTSTOCKCODE9'].iloc[nC]
            ind = BondInfo['INDUSTRYCODE'].iloc[nC]
            df.loc[df['SYMBOL9_STK'] == code, ind] = 1
        # temp = df[industry].sum(axis=1)
        # df2 = df[temp > 1]
    if opt == 'Neut_MKV':
        cols = ['MKV']
    elif opt == 'Neut_IND':
        cols = industry
    elif opt == 'Neut_MI':
        cols = ['MKV'] + industry
    elif opt == 'Neut_MP':
        cols = ['MKV', 'Price']
    else:
        print('opt未定义,', neutralization)

    def group_operation(group):
        # 对每个分组使用statsmodels进行线性回归
        # X = smapi.add_constant(group[cols].values.reshape(-1, 1))  # 添加常数项以拟合截距
        # y = group['FACTORVALUE'].values
        X = group[cols]
        X = smapi.add_constant(X) # 添加常数项
        y = group[col]
        model = smapi.OLS(y, X).fit()
        group[col + opt] = model.resid
        return group
    df = df.groupby('date', group_keys=False).apply(group_operation)

    return df[col + opt]


def plot_cumulative_returns(df_plot, config):
    #
    print('plot_cumulative_returns...')
    cmap = LinearSegmentedColormap.from_list("RedBlue", ["red", "blue"], config['n_quantiles'])
    colors = cmap(np.linspace(0, 1, config['n_quantiles']))
    df_plot.index = df_plot['date']
    # print('plt.figure...')
    plt.figure(figsize=(12, 8))
    for i in range(0, config['n_quantiles']):
        plt.plot(df_plot.index, df_plot['CE_G' + str(i+1)].cumsum(), label=f'G {i+1}', color=colors[i])

    plt.rcParams['font.family'] = 'Arial Unicode MS'
    # print('plot_cumulative_returns, title...')
    plt.title(config['factorname'])
    plt.xlabel('日期')
    plt.ylabel('累计收益')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('plot_cumulative_returns, savefig...')
    out_path = config['plot_dir']
    os.makedirs(out_path, exist_ok=True)
    file_path = os.path.join(out_path, config['factorname'] + '.png')
    plt.savefig(file_path)
    plt.close('all')

    return 0

def plot_monthly_returns(monthly_returns, config):
    #
    print('plot_monthly_returns...')
    monthly_returns.index = monthly_returns['month']
    plt.figure(figsize=(12, 8))
    monthly_returns['return'].plot(kind='bar', color='blue', alpha=0.7)
    plt.title(config['factorname'])
    plt.xlabel('月度')
    plt.ylabel('累计收益')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    print('plot_monthly_returns, savefig...')
    out_path = config['plot_dir']
    os.makedirs(out_path, exist_ok=True)
    file_path = os.path.join(out_path, config['factorname'] + '_month.png')
    plt.savefig(file_path)
    plt.close('all')

    return 0


# 从list中删除指定元素
def remove_value_from_List(lst, value):
    return [x for x in lst if x != value]
    # list1 = [item for item in FactorList if item not in FactorList_MKT]



# 将个债个股信息表拆分成按个股保存
def splitQuoteInfo2stk(config, STKlist, TradeDays):
    folder_path = config['Info_dir'] + config['folder_path'] + '/'
    TradeDays = fetchCalendard2Mth(TradeDays)
    TradeMth = TradeDays['TradeMth'].unique()

    df_all = []
    for tempMth in TradeMth:
        if config['folder_path'] == 'Bond':
            file_path = folder_path + 'Mth/BOND_Quotation_' + str(tempMth) + '.csv'
        elif config['folder_path'] == 'STK_Org':
            file_path = folder_path + 'Mth/STK_MKT_Quotation_' + str(tempMth) + '.csv'
        elif config['folder_path'] == 'STK_Fwd':
            file_path = folder_path + 'Mth/STK_MKT_FwardQuotation_' + str(tempMth) + '.csv'
        # print(file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if len(df_all) == 0:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            print('文件不存在，' + file_path)

    # exchange_dict = {"SSE": "SH", "SZSE": "SZ"}
    # df_all['EXCHANGECODE'] = df_all['EXCHANGECODE'].map(exchange_dict)
    df_all = df_all.drop(['SECURITYID'], axis=1)

    # dfGroup = df_all.groupby(['SYMBOL', 'EXCHANGECODE'])
    dfGroup = df_all.groupby(['SYMBOL9'])
    count = 0
    for name, group in dfGroup:
        count += 1
        # print(name)
        # print(group)
        if count%500 == 1:
            print(count)
            print(time.strftime("%Y%m%d %H%M%S", time.localtime()))
            print(f"Group name: {name}")
            # print(group)

        group = group.reset_index(drop=True)
        # 保存数据
        filename = f"{name[0]}.csv"
        file_path = os.path.join(folder_path, filename)
        group.to_csv(file_path, index=False)
        #
        # STKlist.remove(name[0])
        STKlist = [item for item in STKlist if item != name[0]]


    return STKlist


# 对因子进行标准化处理
def standardization(df, cols, opt):
    if opt == 'rank':
        def group_operation(group):
            return group.rank() / len(group)
    elif opt == 'meanstd':
        def group_operation(group):
            return (group - group.mean()) / group.std()
    else:
        print('opt未定义', opt)
    for col in cols:
        df[col + '_std'] = df.groupby('date')[col].transform(group_operation)

    return df


# 将6位纯数字600001的正股代码转换为9位的600001.SH
def symbol6_to_symbol9(df):
    df.loc[df['SYMBOL'].str[0] == '0', 'EXCHANGECODE'] = 'SZ'
    df.loc[df['SYMBOL'].str[0] == '3', 'EXCHANGECODE'] = 'SZ'
    df.loc[df['SYMBOL'].str[0] == '6', 'EXCHANGECODE'] = 'SH'
    df['SYMBOL9'] = df['SYMBOL'].astype(str) + '.' + df['EXCHANGECODE']

    return df

