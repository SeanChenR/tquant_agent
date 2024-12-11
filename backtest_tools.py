import os
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

from upload_img import upload_file_to_github

# import zipline
from zipline.data import bundles
from zipline.sources.TEJ_Api_Data import get_universe
from zipline.pipeline.filters import StaticAssets

from zipline.finance import slippage, commission
from zipline.api import *
from zipline import run_algorithm

from zipline.pipeline import Pipeline
from zipline.pipeline.factors import IchimokuKinkoHyo, TrueRange, CustomFactor
from zipline.pipeline.data import TWEquityPricing, EquityPricing
from zipline.utils.math_utils import nanmax
from numpy import dstack
import pyfolio as pf

@tool
def Ichimoku_Kinko_Hyo(start, end, idx_id):
    """
    Select the stock universe and ingest the data into zipline and run backtesting with the strategy of 一目均衡表
    """
    StockList = get_universe(start, end, idx_id=idx_id)
    ticker = ','.join(StockList)

    with open('backtest_stats.txt', 'w') as f:
        f.write(f"Start date: {start}\n")
        f.write(f"End date: {end}\n")
        f.write(f"StockList: {ticker}\n")
        f.write(f"Benchmark: IR0001\n")
        f.write(f"Strategy: IchimokuKinkoHyo (一目均衡表)\n")

    StockList.append('IR0001')

    os.environ['ticker'] = ' '.join(StockList)
    os.environ['mdate'] = start+' '+end

    start_dt, end_dt = pd.Timestamp(start, tz='utc'), pd.Timestamp(end, tz='utc')

    # !zipline ingest -b tquant
    command = f"zipline ingest -b tquant"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

    bundle = bundles.load('tquant')
    benchmark_asset = bundle.asset_finder.lookup_symbol('IR0001', as_of_date = None)

    class AverageTrueRange(CustomFactor):

        inputs = (
            EquityPricing.high,
            EquityPricing.low,
            EquityPricing.close,
        )
        
        window_length = 10

        outputs = ["TR", "ATR"]
        
        def compute(self, today, assets, out, highs, lows, closes):

            high_to_low = highs[1:] - lows[1:]
            high_to_prev_close = abs(highs[1:] - closes[:-1])
            low_to_prev_close = abs(lows[1:] - closes[:-1])
            tr_current = nanmax(
                dstack(
                    (
                        high_to_low,
                        high_to_prev_close,
                        low_to_prev_close,
                    )
                ),
                2,
            )

            sma_atr_values = np.mean(tr_current, axis=0)
            
            out.TR = tr_current[-1]
            out.ATR = sma_atr_values

    def make_pipeline():
        
        Ich = IchimokuKinkoHyo(
            inputs = [TWEquityPricing.high, TWEquityPricing.low, TWEquityPricing.close],
            window_length = 52,
        )
        atr = AverageTrueRange(inputs = [TWEquityPricing.high, TWEquityPricing.low, TWEquityPricing.close],
            window_length = 52,
        )
        
        return Pipeline(
            columns = {
                'curr_price': TWEquityPricing.close.latest,
                "tenkan_sen": Ich.tenkan_sen,
                "kijun_sen": Ich.kijun_sen,
                "senkou_span_a": Ich.senkou_span_a,
                "senkou_span_b": Ich.senkou_span_b,
                'cloud_red': Ich.senkou_span_a < Ich.senkou_span_b,
                "chikou_span": Ich.chikou_span,
                'stop_loss': atr.ATR,
        },
            # screen = ~StaticAssets([benchmark_asset])
            screen = ~StaticAssets([benchmark_asset]) & (Ich.senkou_span_a > 0) & (Ich.senkou_span_b > 0)
        )
    # my_pipeline = run_pipeline(make_pipeline(), start_dt, end_dt)

    def initialize(context):
        set_slippage(slippage.VolumeShareSlippage())
        set_commission(commission.Custom_TW_Commission(min_trade_cost = 20, discount = 1.0, tax = 0.003))
        attach_pipeline(make_pipeline(), 'mystrats')
        set_benchmark(symbol('IR0001'))
        context.stop_loss = {}
        context.trailing_stop = {}
        context.last_buy_price = {}
        context.trailing_count = {}
        context.holding = {}
        context.buy_count = {}

    def handle_data(context, data):
        out_dir = pipeline_output('mystrats')

        for i in out_dir.index:
            sym = i.symbol
            curr_price = out_dir.loc[i, 'curr_price']
            tenkan_sen = out_dir.loc[i, 'tenkan_sen']
            kijun_sen = out_dir.loc[i, 'kijun_sen']
            senkou_span_a = out_dir.loc[i, 'senkou_span_a']
            senkou_span_b = out_dir.loc[i, 'senkou_span_b']
            cloud_red = out_dir.loc[i, 'cloud_red']
            chikou_span = out_dir.loc[i, 'chikou_span']
            stop_loss = out_dir.loc[i, 'stop_loss']
            cash_position = context.portfolio.cash  # record cash position
            stock_position = context.portfolio.positions[i].amount  # record stock holding

            if context.stop_loss.get(f'{i}') is None:
                context.stop_loss[f'{i}'] = 0
                
            if context.trailing_stop.get(f'{i}') is None:
                context.trailing_stop[f'{i}'] = False
                
            if context.last_buy_price.get(f'{i}') is None:
                context.last_buy_price[f'{i}'] = 0

            if context.holding.get(f'{i}') is None:
                context.holding[f'{i}'] = False
                
            if context.trailing_count.get(f'{i}') is None:
                context.trailing_count[f'{i}'] = 1

            if context.buy_count.get(f'{i}') is None:
                context.buy_count[f'{i}'] = 0
                
            buy, sell = False, False
            record(
            **{
                    f'price_{sym}':curr_price,
                    f'buy_{sym}':buy,
                    f'sell_{sym}':sell,
                    f'tenkan_sen_{sym}': tenkan_sen,
                    f'kijun_sen_{sym}': kijun_sen,
                    f'cloud_red_{sym}': cloud_red,
                    f'senkou_span_a_{sym}': senkou_span_a,
                    f'senkou_span_b_{sym}': senkou_span_b,
                    f'chikou_span_{sym}': chikou_span,
                }
            )
            
            # 三役好轉 (tenkan_sen > kijun_sen*1.015 : avoid the Darvas Box Theory)
            if (curr_price > senkou_span_b) and (cloud_red == True) and (tenkan_sen > kijun_sen*1.01) and (context.buy_count[f'{i}'] <= 5):
                order_percent(i, 0.01)
                buy = True
                context.stop_loss[f'{i}'] = curr_price - (1.25 * stop_loss)
                context.last_buy_price[f'{i}'] = curr_price
                record(
                    **{
                        f'buy_{sym}':buy
                    }
                )
                context.holding[f'{i}'] = True
                context.buy_count[f'{i}'] += 1

            # reset stop loss point
            if (curr_price >= (1.3**context.trailing_count[f'{i}'])*context.last_buy_price[f'{i}']) and (context.holding[f'{i}'] == True) and (context.trailing_stop[f'{i}'] == False):
                context.stop_loss[f'{i}'] = 1.3*context.stop_loss[f'{i}']
                context.trailing_stop[f'{i}'] = True
                context.trailing_count[f'{i}'] += 1
            elif (curr_price >= (1.3**context.trailing_count[f'{i}'])*context.last_buy_price[f'{i}']) and (context.holding[f'{i}'] == True) and (context.trailing_stop[f'{i}'] == True):
                context.stop_loss[f'{i}'] = 1.3*context.stop_loss[f'{i}']
                context.trailing_count[f'{i}'] += 1
            
            if (curr_price <= context.stop_loss[f'{i}']) and (context.holding[f'{i}'] == True):
                order_target(i, 0)
                sell = True
                context.stop_loss[f'{i}'] = None
                context.trailing_stop[f'{i}'] = None
                context.trailing_count[f'{i}'] = None
                record(
                    **{
                        f'sell_{sym}':sell
                    }
                )
                context.holding[f'{i}'] = None
                context.buy_count[f'{i}'] = None

    results = run_algorithm(
        start = start_dt,
        end = end_dt,
        initialize = initialize,
        bundle = 'tquant',
        capital_base = 1e7,
        handle_data = handle_data
    )

    bt_returns, bt_positions, bt_transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)
    benchmark_rets = results.benchmark_return

    perf_stats = pf.plotting.show_perf_stats(
        bt_returns, 
        benchmark_rets, 
        bt_positions, 
        bt_transactions, 
        turnover_denom='portfolio_value',
    )

    # 打開一個txt文件，並將資料寫入
    with open('backtest_stats.txt', 'a') as f:
        for index, row in perf_stats.iterrows():
            # 將每個指標和對應的值寫入文本文件
            f.write(f"{index}: {row[0]}\n")

    with open("backtest_img_url.txt", "r+") as file:  # 'r+' 模式表示可讀寫
        file.truncate(0)

    # Cumulative Returns
    pf.plotting.plot_rolling_returns(bt_returns, benchmark_rets)
    plt.savefig('image/cumulative_returns.png')
    plt.close()
    upload_file_to_github('image/cumulative_returns.png')

    # Rolling Volatility
    pf.plotting.plot_rolling_volatility(bt_returns, benchmark_rets)
    plt.savefig('image/rolling_volatility.png')
    plt.close()
    upload_file_to_github('image/rolling_volatility.png')

    # Rolling Sharpe
    pf.plotting.plot_rolling_sharpe(bt_returns, benchmark_rets)
    plt.savefig('image/rolling_sharpe.png')
    plt.close()
    upload_file_to_github('image/rolling_sharpe.png')

    # drawdown
    pf.plotting.plot_drawdown_underwater(bt_returns)
    plt.savefig('image/drawdown.png')
    plt.close()
    upload_file_to_github('image/drawdown.png')

    # monthly returns heatmap
    pf.plotting.plot_monthly_returns_heatmap(bt_returns)
    plt.savefig('image/monthly_returns_heatmap.png')
    plt.close()
    upload_file_to_github('image/monthly_returns_heatmap.png')

    # annual returns
    pf.plotting.plot_annual_returns(bt_returns)
    plt.savefig('image/annual_returns.png')
    plt.close()
    upload_file_to_github('image/annual_returns.png')

    # exposures
    pf.plotting.plot_exposures(bt_returns, bt_positions)
    plt.savefig('image/exposures.png')
    plt.close()
    upload_file_to_github('image/exposures.png')

    # Long & Short Holdings
    pf.plotting.plot_long_short_holdings(bt_returns, bt_positions)
    plt.savefig('image/long_short_holdings.png')
    plt.close()
    upload_file_to_github('image/long_short_holdings.png')

    # turnover
    pf.plotting.plot_turnover(bt_returns, bt_transactions, bt_positions)
    plt.savefig('image/turnover.png')
    plt.close()
    upload_file_to_github('image/turnover.png')

    # daily volume
    pf.plotting.plot_daily_volume(bt_returns, bt_transactions)
    plt.savefig('image/daily_volume.png')
    plt.close()
    upload_file_to_github('image/daily_volume.png')

    with open("backtest_stats.txt", "r") as file:
        content = file.read()
    
    return content

# # 抓取股票範圍
# start = '2019-06-01'
# end = '2024-06-01'
# start_dt, end_dt = pd.Timestamp(start, tz='utc'), pd.Timestamp(end, tz='utc')

# # 抓取台灣50指數的股票
# StockList = get_universe(start, end, idx_id='IX0002')
# StockList.append('IR0001')

# os.environ['ticker'] = ' '.join(StockList)
# os.environ['mdate'] = start+' '+end

# # !zipline ingest -b tquant
# command = f"zipline ingest -b tquant"
# try:
#     result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print(result.stdout.decode())
# except subprocess.CalledProcessError as e:
#     print(f"Error occurred: {e.stderr.decode()}")

# bundle = bundles.load('tquant')
# benchmark_asset = bundle.asset_finder.lookup_symbol('IR0001',as_of_date = None)

# class AverageTrueRange(CustomFactor):

#     inputs = (
#         EquityPricing.high,
#         EquityPricing.low,
#         EquityPricing.close,
#     )
    
#     window_length = 10

#     outputs = ["TR", "ATR"]
    
#     def compute(self, today, assets, out, highs, lows, closes):

#         high_to_low = highs[1:] - lows[1:]
#         high_to_prev_close = abs(highs[1:] - closes[:-1])
#         low_to_prev_close = abs(lows[1:] - closes[:-1])
#         tr_current = nanmax(
#             dstack(
#                 (
#                     high_to_low,
#                     high_to_prev_close,
#                     low_to_prev_close,
#                 )
#             ),
#             2,
#         )

#         sma_atr_values = np.mean(tr_current, axis=0)
        
#         out.TR = tr_current[-1]
#         out.ATR = sma_atr_values

# def make_pipeline():
    
#     Ich = IchimokuKinkoHyo(
#         inputs = [TWEquityPricing.high, TWEquityPricing.low, TWEquityPricing.close],
#         window_length = 52,
#     )
#     atr = AverageTrueRange(inputs = [TWEquityPricing.high, TWEquityPricing.low, TWEquityPricing.close],
#         window_length = 52,
#     )
    
#     return Pipeline(
#         columns = {
#             'curr_price': TWEquityPricing.close.latest,
#             "tenkan_sen": Ich.tenkan_sen,
#             "kijun_sen": Ich.kijun_sen,
#             "senkou_span_a": Ich.senkou_span_a,
#             "senkou_span_b": Ich.senkou_span_b,
#             'cloud_red': Ich.senkou_span_a < Ich.senkou_span_b,
#             "chikou_span": Ich.chikou_span,
#             'stop_loss': atr.ATR,
#     },
#         # screen = ~StaticAssets([benchmark_asset])
#         screen = ~StaticAssets([benchmark_asset]) & (Ich.senkou_span_a > 0) & (Ich.senkou_span_b > 0)
#     )
# my_pipeline = run_pipeline(make_pipeline(), start_dt, end_dt)

# def initialize(context):
#     set_slippage(slippage.VolumeShareSlippage())
#     set_commission(commission.Custom_TW_Commission(min_trade_cost = 20, discount = 1.0, tax = 0.003))
#     attach_pipeline(make_pipeline(), 'mystrats')
#     set_benchmark(symbol('IR0001'))
#     context.stop_loss = {}
#     context.trailing_stop = {}
#     context.last_buy_price = {}
#     context.trailing_count = {}
#     context.holding = {}
#     context.buy_count = {}

# def handle_data(context, data):
#     out_dir = pipeline_output('mystrats')

#     for i in out_dir.index:
#         sym = i.symbol
#         curr_price = out_dir.loc[i, 'curr_price']
#         tenkan_sen = out_dir.loc[i, 'tenkan_sen']
#         kijun_sen = out_dir.loc[i, 'kijun_sen']
#         senkou_span_a = out_dir.loc[i, 'senkou_span_a']
#         senkou_span_b = out_dir.loc[i, 'senkou_span_b']
#         cloud_red = out_dir.loc[i, 'cloud_red']
#         chikou_span = out_dir.loc[i, 'chikou_span']
#         stop_loss = out_dir.loc[i, 'stop_loss']
#         cash_position = context.portfolio.cash  # record cash position
#         stock_position = context.portfolio.positions[i].amount  # record stock holding

#         if context.stop_loss.get(f'{i}') is None:
#             context.stop_loss[f'{i}'] = 0
            
#         if context.trailing_stop.get(f'{i}') is None:
#             context.trailing_stop[f'{i}'] = False
            
#         if context.last_buy_price.get(f'{i}') is None:
#             context.last_buy_price[f'{i}'] = 0

#         if context.holding.get(f'{i}') is None:
#             context.holding[f'{i}'] = False
            
#         if context.trailing_count.get(f'{i}') is None:
#             context.trailing_count[f'{i}'] = 1

#         if context.buy_count.get(f'{i}') is None:
#             context.buy_count[f'{i}'] = 0
            
#         buy, sell = False, False
#         record(
#            **{
#                 f'price_{sym}':curr_price,
#                 f'buy_{sym}':buy,
#                 f'sell_{sym}':sell,
#                 f'tenkan_sen_{sym}': tenkan_sen,
#                 f'kijun_sen_{sym}': kijun_sen,
#                 f'cloud_red_{sym}': cloud_red,
#                 f'senkou_span_a_{sym}': senkou_span_a,
#                 f'senkou_span_b_{sym}': senkou_span_b,
#                 f'chikou_span_{sym}': chikou_span,
#             }
#         )
        
#         # 三役好轉 (tenkan_sen > kijun_sen*1.015 : avoid the Darvas Box Theory)
#         if (curr_price > senkou_span_b) and (cloud_red == True) and (tenkan_sen > kijun_sen*1.01) and (context.buy_count[f'{i}'] <= 5):
#             order_percent(i, 0.01)
#             buy = True
#             context.stop_loss[f'{i}'] = curr_price - (1.25 * stop_loss)
#             context.last_buy_price[f'{i}'] = curr_price
#             record(
#                 **{
#                     f'buy_{sym}':buy
#                 }
#             )
#             context.holding[f'{i}'] = True
#             context.buy_count[f'{i}'] += 1

#         # reset stop loss point
#         if (curr_price >= (1.3**context.trailing_count[f'{i}'])*context.last_buy_price[f'{i}']) and (context.holding[f'{i}'] == True) and (context.trailing_stop[f'{i}'] == False):
#             context.stop_loss[f'{i}'] = 1.3*context.stop_loss[f'{i}']
#             context.trailing_stop[f'{i}'] = True
#             context.trailing_count[f'{i}'] += 1
#         elif (curr_price >= (1.3**context.trailing_count[f'{i}'])*context.last_buy_price[f'{i}']) and (context.holding[f'{i}'] == True) and (context.trailing_stop[f'{i}'] == True):
#             context.stop_loss[f'{i}'] = 1.3*context.stop_loss[f'{i}']
#             context.trailing_count[f'{i}'] += 1
        
#         if (curr_price <= context.stop_loss[f'{i}']) and (context.holding[f'{i}'] == True):
#             order_target(i, 0)
#             sell = True
#             context.stop_loss[f'{i}'] = None
#             context.trailing_stop[f'{i}'] = None
#             context.trailing_count[f'{i}'] = None
#             record(
#                 **{
#                     f'sell_{sym}':sell
#                 }
#             )
#             context.holding[f'{i}'] = None
#             context.buy_count[f'{i}'] = None

# results = run_algorithm(
#     start = start_dt,
#     end = end_dt,
#     initialize = initialize,
#     bundle = 'tquant',
#     capital_base = 1e7,
#     handle_data = handle_data
# )

# bt_returns, bt_positions, bt_transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)
# benchmark_rets = results.benchmark_return

# perf_stats = pf.plotting.show_perf_stats(
#     bt_returns, 
#     benchmark_rets, 
#     bt_positions, 
#     bt_transactions, 
#     turnover_denom='portfolio_value',
# )

# # 打開一個txt文件，並將資料寫入
# with open('backtest_stats.txt', 'w') as f:
#     for index, row in perf_stats.iterrows():
#         # 將每個指標和對應的值寫入文本文件
#         f.write(f"{index}: {row[0]}\n")

# # Cumulative Returns
# pf.plotting.plot_rolling_returns(bt_returns, benchmark_rets)
# plt.savefig('image/cumulative_returns.png')
# plt.close()
# upload_file_to_github('image/cumulative_returns.png')

# # Rolling Volatility
# pf.plotting.plot_rolling_volatility(bt_returns, benchmark_rets)
# plt.savefig('image/rolling_volatility.png')
# plt.close()
# upload_file_to_github('image/rolling_volatility.png')

# # Rolling Sharpe
# pf.plotting.plot_rolling_sharpe(bt_returns, benchmark_rets)
# plt.savefig('image/rolling_sharpe.png')
# plt.close()
# upload_file_to_github('image/rolling_sharpe.png')

# # drawdown
# pf.plotting.plot_drawdown_underwater(bt_returns)
# plt.savefig('image/drawdown.png')
# plt.close()
# upload_file_to_github('image/drawdown.png')

# # monthly returns heatmap
# pf.plotting.plot_monthly_returns_heatmap(bt_returns)
# plt.savefig('image/monthly_returns_heatmap.png')
# plt.close()
# upload_file_to_github('image/monthly_returns_heatmap.png')

# # annual returns
# pf.plotting.plot_annual_returns(bt_returns)
# plt.savefig('image/annual_returns.png')
# plt.close()
# upload_file_to_github('image/annual_returns.png')

# # exposures
# pf.plotting.plot_exposures(bt_returns, bt_positions)
# plt.savefig('image/exposures.png')
# plt.close()
# upload_file_to_github('image/exposures.png')

# # Long & Short Holdings
# pf.plotting.plot_long_short_holdings(bt_returns, bt_positions)
# plt.savefig('image/long_short_holdings.png')
# plt.close()
# upload_file_to_github('image/long_short_holdings.png')

# # turnover
# pf.plotting.plot_turnover(bt_returns, bt_transactions, bt_positions)
# plt.savefig('image/turnover.png')
# plt.close()
# upload_file_to_github('image/turnover.png')

# # daily volume
# pf.plotting.plot_daily_volume(bt_returns, bt_transactions)
# plt.savefig('image/daily_volume.png')
# plt.close()
# upload_file_to_github('image/daily_volume.png')