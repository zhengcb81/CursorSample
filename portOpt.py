import os
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Agg
matplotlib.interactive(True)  # Disable interactive mode
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries

# ... (保留现有的导入语句) ...

# 添加新的资产类层次结构
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, Optional

class Asset(ABC):
    """Base Asset Class"""
    
    def __init__(self, ticker: str, name: str = None):
        self.ticker = ticker
        self.name = name or ticker
        self._data = None
        
    def fetch_data(self, start_date: str, end_date: str, data_source: str = 'alpha_vantage', api_key: str = None) -> pd.DataFrame:
        """Fetch historical data for the asset
        
        Args:
            start_date: Start date
            end_date: End date 
            data_source: Data source ('alpha_vantage' or 'pandas_reader')
            api_key: Alpha Vantage API key
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            if data_source == 'alpha_vantage':
                if not api_key:
                    raise ValueError("API key required when using Alpha Vantage")
                ts = TimeSeries(key=api_key, output_format='pandas')
                data, _ = ts.get_daily_adjusted(symbol=self.ticker, outputsize='full')
                data = data.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. adjusted close': 'Adj Close',
                    '6. volume': 'Volume'
                })
                data = data.loc[start_date:end_date]
            else:
                data = web.DataReader(
                    self.ticker,
                    'stooq',
                    start=start_date,
                    end=end_date
                )
            
            self._data = data
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {self.name}: {str(e)}")
    
    def prices(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get price data for a specific time period"""
        if self._data is None:
            raise ValueError("Please call fetch_data first to get data")
        
        data = self._data
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data

class Equity(Asset):
    """Equity Class"""
    
    def __init__(self, ticker: str, name: str = None, sector: str = None):
        super().__init__(ticker, name)
        self.sector = sector
        self.asset_type = 'equity'
    
    def get_beta(self, market_index: str = 'SPY', data_source: str = 'alpha_vantage', api_key: str = None) -> float:
        """Calculate beta coefficient relative to market"""
        if self._data is None:
            raise ValueError("Please call fetch_data first to get data")
            
        try:
            # Get market index data
            market = Asset(market_index, "Market Index")
            market_data = market.fetch_data(
                start_date=self._data.index[0].strftime('%Y-%m-%d'),
                end_date=self._data.index[-1].strftime('%Y-%m-%d'),
                data_source=data_source,
                api_key=api_key
            )
            
            # Calculate returns
            stock_returns = self._data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Ensure data alignment
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            
            # Calculate beta coefficient
            covariance = aligned_data.cov().iloc[0,1]
            market_variance = aligned_data.iloc[:,1].var()
            beta = covariance / market_variance
            
            return beta
            
        except Exception as e:
            raise ValueError(f"Failed to calculate beta: {str(e)}")

class Bond(Asset):
    """Bond Class"""
    
    def __init__(self, ticker: str, name: str = None, duration: float = None):
        super().__init__(ticker, name)
        self.duration = duration
        self.asset_type = 'bond'
    
    def get_yield(self) -> float:
        """Get bond yield"""
        if self._data is None:
            raise ValueError("Please call fetch_data first to get data")
        
        # Calculate simple yield using latest adjusted close price
        latest_price = self._data['Adj Close'][-1]
        year_ago_price = self._data['Adj Close'][-252] if len(self._data) >= 252 else self._data['Adj Close'][0]
        simple_yield = (latest_price / year_ago_price - 1) * 100
        return simple_yield

class FX(Asset):
    """Foreign Exchange Class"""
    
    def __init__(self, ticker: str, name: str = None, base_currency: str = None, quote_currency: str = None):
        super().__init__(ticker, name)
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.asset_type = 'fx'
    
    def get_volatility(self, window: int = 252) -> float:
        """Calculate exchange rate volatility"""
        if self._data is None:
            raise ValueError("Please call fetch_data first to get data")
            
        returns = self._data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(window)
        return volatility

class MarketEnvironment:
    """Market Environment class for fetching and managing market data"""
    
    def __init__(self, assets: List[Asset], start_date: str, end_date: str, 
                 data_source: str = 'alpha_vantage', debug: bool = False):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source
        self.debug = debug
        self.api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Need to replace with actual API key
        self.data = self._fetch_data()
        if self.debug:
            self.visualize_data()
        
    def visualize_data(self) -> None:
        """Visualize market data"""
        # Plot price trends
        plt.figure(figsize=(12, 6))
        for asset in self.assets:
            plt.plot(self.data.index, self.data[asset.ticker], label=asset.name)
        plt.title('Asset Price Trends')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Asset Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Plot volatility comparison
        volatility = self.data.pct_change().std() * np.sqrt(252)
        plt.figure(figsize=(10, 6))
        volatility.plot(kind='bar')
        plt.title('Annual Asset Volatility')
        plt.xlabel('Asset')
        plt.ylabel('Volatility')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_returns(self, debug: bool = False) -> pd.DataFrame:
        """计算资产的每日收益率
        
        Args:
            debug: 是否显示可视化图表，默认为False
        
        Returns:
            pd.DataFrame: 包含所有资产每日收益率的DataFrame
        """
        # 计算收益率
        returns = self.data.pct_change()
        
        if debug:
            # 绘制收益率分布图
            plt.figure(figsize=(12, 6))
            returns.hist(bins=50, alpha=0.5)
            plt.title('资产收益率分布')
            plt.xlabel('收益率')
            plt.ylabel('频率')
            plt.grid(True)
            plt.show()
            
            # 绘制收益率时间序列图
            plt.figure(figsize=(12, 6))
            returns.plot()
            plt.title('资产收益率时间序列')
            plt.xlabel('日期')
            plt.ylabel('收益率')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        returns = returns.dropna()
        return returns

    def _fetch_data_alpha_vantage(self, asset) -> pd.DataFrame:
        """Fetch data using Alpha Vantage API"""
        try:
            ts = TimeSeries(key=self.api_key, output_format='pandas')
            data, _ = ts.get_daily_adjusted(symbol=asset.ticker, outputsize='full')
            # Rename columns to match our format
            data = data.rename(columns={'4. close': 'Close'})
            # Select date range
            data = data.loc[self.start_date:self.end_date]
            return data
        except Exception as e:
            print(f"Alpha Vantage API Error: {str(e)}")
            return None

    def _fetch_data_pandas_reader(self, asset) -> pd.DataFrame:
        """Fetch data using pandas_datareader"""
        try:
            data = web.DataReader(
                asset.ticker,
                'stooq',  # Use stooq as data source
                start=self.start_date,
                end=self.end_date
            )
            return data
        except Exception as e:
            print(f"Pandas DataReader Error: {str(e)}")
            return None

    def _fetch_data(self) -> pd.DataFrame:
        """Fetch historical data for all assets"""
        try:
            data = pd.DataFrame()
            max_retries = 3

            for asset in self.assets:
                success = False
                retries = 0

                while not success and retries < max_retries:
                    try:
                        # Get data based on selected data source
                        if self.data_source == 'alpha_vantage':
                            asset_data = self._fetch_data_alpha_vantage(asset)
                        else:
                            asset_data = self._fetch_data_pandas_reader(asset)

                        if asset_data is None or asset_data.empty:
                            retries += 1
                            print(f"Retry {retries}/{max_retries} - {asset.name}")
                            time.sleep(2)
                            continue

                        data[asset.ticker] = asset_data['Close']
                        success = True
                        print(f"Successfully retrieved historical data for {asset.name}")

                    except Exception as e:
                        retries += 1
                        print(f"Error getting data for {asset.name}: {str(e)}")
                        if retries < max_retries:
                            print(f"Waiting to retry ({retries}/{max_retries})...")
                            time.sleep(2)
                        else:
                            print(f"Unable to get data for {asset.name}, maximum retries reached")
                            raise ValueError(f"Unable to get data for {asset.name}")

            # Data validation and cleaning
            if data.empty:
                raise ValueError("No asset data retrieved")

            # 确保数据按日期升序排列
            data = data.sort_index(ascending=True)

            # 检查并处理缺失值
            missing_data = data.isnull().sum()
            if missing_data.any():
                print("Warning: Missing values in data:")
                print(missing_data[missing_data > 0])
                data = data.fillna(method='ffill').fillna(method='bfill')
                print("Missing data filled using nearby values")

            return data

        except Exception as e:
            raise ValueError(f"Error occurred while fetching data: {str(e)}")
class PortfolioOptimizer:
    """Portfolio Optimizer"""
    
    def __init__(self, market_env: MarketEnvironment):
        self.market_env = market_env
        self.returns = market_env.get_returns()
        
    def optimize_sharpe(self, risk_free_rate: float = 0.02, window_days: int = 30, current_date: pd.Timestamp = None) -> Dict:
        """基于定时间窗口优化投资组合权重
        
        Args:
            risk_free_rate: 无风险利率
            window_days: 回溯时间窗口天数
            current_date: 当前日期，如果为None则使用数据中的最后一个日期
            
        Returns:
            Dict: 优化后的资产权重字典
        """
        if current_date is None:
            current_date = self.returns.index[-1]
            
        # 获取时间窗口内的收益率数据
        end_date = current_date
        start_date = current_date - pd.Timedelta(days=window_days)
        window_returns = self.returns[
            (self.returns.index >= start_date) & 
            (self.returns.index <= end_date)
        ]
        
        # 检查是否有足够的数据
        if len(window_returns) < 5:  # 至少需要5个交易日的数据
            raise ValueError(f"不够的数据点用于优化 (got {len(window_returns)})")
            
        n_assets = len(self.market_env.assets)
        
        def objective(weights):
            portfolio_return = np.sum(window_returns.mean() * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(window_returns.cov() * 252, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # 最小化负夏普比率 = 最大化夏普比率
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 权重和为1
        bounds = tuple((0, 1) for _ in range(n_assets))  # 权重在0和1之间
        
        try:
            result = minimize(objective, 
                            x0=np.array([1/n_assets] * n_assets),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if not result.success:
                print(f"优化警告: {result.message}")
                
            # 返回优化结果
            return dict(zip([asset.ticker for asset in self.market_env.assets], result.x))
            
        except Exception as e:
            print(f"优化错误: {str(e)}")
            # 如果优化失败，返回平均分配的权重
            return dict(zip([asset.ticker for asset in self.market_env.assets], 
                          [1/n_assets] * n_assets))

class BackTester:
    """Backtesting Framework"""
    
    def __init__(self, market_env: MarketEnvironment, optimizer: PortfolioOptimizer):
        self.market_env = market_env
        self.optimizer = optimizer
        
    def run_backtest(self, rebalance_frequency: str = '1M') -> pd.DataFrame:
        """执行回测并回权重和收益的时间序列
        
        Args:
            rebalance_frequency: 再平衡频率 ('1D' 每日, '1W' 每周, '1M' 每月等)
            
        Returns:
            pd.DataFrame: 包含每日收益和资产权重的DataFrame
        """
        returns = self.market_env.get_returns()
        
        # Add error checking
        if returns.empty:
            raise ValueError("Returns data is empty. Please check if market data was retrieved correctly.")
            
        portfolio_data = []
        window_days = 30
        asset_tickers = [asset.ticker for asset in self.market_env.assets]

        # 确保开始日期有足够的历史数据用于优化
        start_date = returns.index[0] + pd.Timedelta(days=window_days)
        
        for date in pd.date_range(start=start_date, 
                                end=returns.index[-1], 
                                freq=rebalance_frequency):
            try:
                # 使用30天窗口进行优化
                current_weights = self.optimizer.optimize_sharpe(
                    window_days=window_days,
                    current_date=date
                )
                
                # 计算当日收益
                if date in returns.index:
                    daily_return = sum(returns.loc[date] * list(current_weights.values()))
                    
                    # 创建包含所有信息的记录
                    record = {
                        'Date': date,
                        'Returns': daily_return
                    }
                    # 添加每个资产的权重
                    for ticker, weight in current_weights.items():
                        record[f'{ticker}_weight'] = weight
                        
                    portfolio_data.append(record)
                    
            except Exception as e:
                print(f"Error on date {date}: {str(e)}")
                continue
        
        # 转换为DataFrame
        results = pd.DataFrame(portfolio_data)
        results.set_index('Date', inplace=True)
        
        # 确保按日期升序排列
        results = results.sort_index(ascending=True)
        
        # 分离权重列和收益列
        weight_cols = [col for col in results.columns if col.endswith('_weight')]
        
        # 打印回测统计信息
        print("\n=== 回测统计 ===")
        print(f"回测期间: {results.index[0]} 到 {results.index[-1]}")
        print(f"再平衡频率: {rebalance_frequency}")
        print(f"总交易日数: {len(results)}")
        
        # 计算并打印基本指标
        cumulative_return = (1 + results['Returns']).cumprod().iloc[-1] - 1
        annual_return = results['Returns'].mean() * 252
        annual_vol = results['Returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol
        
        print(f"\n累计收益率: {cumulative_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"年化波动率: {annual_vol:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        
        return results

class PortfolioVisualizer:
    """Portfolio Visualization Tool"""
    
    def __init__(self, backtester: BackTester):
        self.backtester = backtester
        plt.style.use('default')
        # Create directory for saving plots
        self.save_dir = 'portfolio_plots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def _save_plot(self, filename: str) -> None:
        """Save plot to local drive
        
        Args:
            filename: Name of the file
        """
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
    
    def plot_cumulative_returns(self, results: pd.DataFrame) -> None:
        """Plot portfolio cumulative returns"""
        plt.figure(figsize=(12, 6))
        cumulative_returns = (1 + results['Returns']).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, label='Portfolio Cumulative Returns')
        plt.title('Portfolio Cumulative Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()
        self._save_plot('cumulative_returns.png')
        plt.close()
    
    def plot_asset_weights(self, weights: Dict) -> None:
        """Plot asset allocation weights pie chart"""
        plt.figure(figsize=(10, 8))
        labels = [asset.name for asset in weights.keys()]
        sizes = [weight * 100 for weight in weights.values()]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Portfolio Asset Allocation')
        plt.axis('equal')
        plt.show()
        self._save_plot('asset_weights.png')
        plt.close()
    
    def plot_rolling_metrics(self, results: pd.DataFrame, window: int = 252) -> None:
        """Plot rolling risk metrics"""
        rolling_vol = results['Returns'].rolling(window=window).std() * np.sqrt(252)
        rolling_return = results['Returns'].rolling(window=window).mean() * 252
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(rolling_return.index, rolling_return.values, label='Annual Return')
        ax1.set_title('Rolling Annual Return')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(rolling_vol.index, rolling_vol.values, label='Annual Volatility', color='orange')
        ax2.set_title('Rolling Annual Volatility')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        self._save_plot('rolling_metrics.png')
        plt.close()

    def plot_weights_timeline(self, weights_history: Dict[str, pd.Series]) -> None:
        """Plot asset weights evolution over time
        
        Args:
            weights_history: Dictionary containing weight series for each asset
        """
        plt.figure(figsize=(12, 6))
        
        # Convert dictionary to DataFrame for easier plotting
        weights_df = pd.DataFrame(weights_history)
        
        # Create stacked area plot
        ax = weights_df.plot(
            kind='area', 
            stacked=True,
            alpha=0.8,
            linewidth=0
        )
        
        plt.title('Portfolio Weights Evolution')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(title='Assets', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Add percentage labels
        for i in range(len(weights_df.columns)):
            y_pos = weights_df.iloc[:, :i+1].sum(axis=1)
            plt.plot(weights_df.index, y_pos, color='white', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        self._save_plot('weights_timeline.png')
        plt.close()

    def plot_weights_heatmap(self, weights_history: Dict[str, pd.Series]) -> None:
        """Plot asset weights heatmap over time
        
        Args:
            weights_history: Dictionary containing weight series for each asset
        """
        plt.figure(figsize=(12, 8))
        
        # Convert dictionary to DataFrame
        weights_df = pd.DataFrame(weights_history)
        
        # Create heatmap
        sns.heatmap(
            weights_df.T,  # Transpose for better visualization
            cmap='YlOrRd',
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Weight'},
            xticklabels=20  # Show fewer x-axis labels for clarity
        )
        
        plt.title('Portfolio Weights Heatmap')
        plt.xlabel('Date')
        plt.ylabel('Asset')
        
        plt.tight_layout()
        plt.show()
        self._save_plot('weights_heatmap.png')
        plt.close()

# Main program example
if __name__ == "__main__":
    # Create different types of assets
    bond = Bond("AGG", name="US Bond ETF", duration=6.7)
    equity1 = Equity("SPY", name="S&P 500 ETF", sector="Large Cap")
    equity2 = Equity("QQQ", name="NASDAQ ETF", sector="Technology")
    fx = FX("EURUSD=X", name="EUR/USD", base_currency="EUR", quote_currency="USD")
    
    assets = [bond, equity1, equity2] #fx
    
    # Create market environment
    market_env = MarketEnvironment(
        assets=assets,
        start_date='2020-01-01',
        end_date='2023-12-31',
        data_source='pandas_reader',  #'alpha_vantage'
        debug=False  # 设置为True时会显示数据可视化
    )

    # Create optimizer
    optimizer = PortfolioOptimizer(market_env)

    # Create backtester
    backtester = BackTester(market_env, optimizer)

    # Run backtest
    results = backtester.run_backtest(rebalance_frequency='1M')

    # Create visualizer
    visualizer = PortfolioVisualizer(backtester)

    # Show various visualizations
    visualizer.plot_cumulative_returns(results)
    
    # 获取权重相关的列
    weight_cols = [col for col in results.columns if col.endswith('_weight')]
    weights_history = {col.replace('_weight', ''): results[col] for col in weight_cols}
    
    # Plot weights timeline and heatmap
    visualizer.plot_weights_timeline(weights_history)
    visualizer.plot_weights_heatmap(weights_history)

    # Print final performance statistics
    final_value = (1 + results['Returns']).cumprod().iloc[-1]
    annual_return = results['Returns'].mean() * 252
    annual_vol = results['Returns'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol
    
    print("\n=== Portfolio Performance Statistics ===")
    print(f"Cumulative Return: {(final_value - 1):.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # 打印最终权重分配
    print("\n=== Final Portfolio Weights ===")
    final_weights = results[weight_cols].iloc[-1]
    for asset, weight in final_weights.items():
        print(f"{asset.replace('_weight', '')}: {weight:.2%}")


