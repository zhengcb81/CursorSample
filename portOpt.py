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
from abc import ABC, abstractmethod
import logging
import unittest
import functools
import yaml

# ... (保留现有的导入语句) ...

# 添加新的资产类层次结构
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, Optional

class PortfolioLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
        
        # Create file handler
        log_file = os.path.join(log_dir, f'{name}.log')
        try:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            
            # Create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            
            self.logger.info(f"Logger initialized. Log file: {log_file}")
            
        except Exception as e:
            print(f"Error setting up logger: {str(e)}")
            raise

class PortfolioError(Exception):
    """Base exception class for portfolio operations"""
    pass

class DataFetchError(PortfolioError):
    """Error when fetching market data"""
    pass

class OptimizationError(PortfolioError):
    """Error during portfolio optimization"""
    pass

class PortfolioMonitor:
    def __init__(self):
        self.metrics = {}
        self.logger = PortfolioLogger("PortfolioMonitor")
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((timestamp, value))
        self.logger.logger.debug(f"Recorded metric {name}: {value}")
    
    def get_metric_history(self, name: str) -> List[tuple]:
        return self.metrics.get(name, [])

class PortfolioConfig:
    def __init__(self, config_file: str = 'portfolio_config.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config file: {str(e)}")
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation
        
        Example:
            config.get('data.source')
            config.get('optimization.risk_free_rate')
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def get_assets(self) -> List[Dict]:
        """Get list of assets from config"""
        return self.config.get('assets', [])
        
    def get_optimization_params(self) -> Dict:
        """Get optimization parameters"""
        return self.config.get('optimization', {})
        
    def get_backtest_params(self) -> Dict:
        """Get backtest parameters"""
        return self.config.get('backtest', {})

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = PortfolioConfig('test_config.yaml')
        self.monitor = PortfolioMonitor()
        
    def test_optimization(self):
        # Test optimization logic
        pass
    
    def test_data_fetching(self):
        # Test data fetching
        pass

def performance_monitor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {duration:.2f} seconds to execute")
        return result
    return wrapper

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
                 data_source: str = 'yfinance', debug: bool = False):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source
        self.debug = debug
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
        """Calculate daily returns for assets
        
        Args:
            debug: Whether to display visualization charts, default False
        
        Returns:
            pd.DataFrame: DataFrame containing daily returns for all assets
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
            
            # 绘制收益率时间序列
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

            # 确保数据按序排列
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

# 优化策略的抽取
class OptimizationStrategy(ABC):
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, risk_free_rate: float) -> Dict:
        """优化投资组合权重的抽象方法"""
        pass

# 夏普比率优化策略
class SharpeOptimizationStrategy(OptimizationStrategy):
    def optimize(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
        """基于夏普比率优化投资组合
        
        Args:
            returns: 收益率数据
            risk_free_rate: 无风险利率
            
        Returns:
            Dict: 优化后的资产权重字典
        """
        n_assets = returns.shape[1]
        
        def objective(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        try:
            result = minimize(objective, 
                            x0=np.array([1/n_assets] * n_assets),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if not result.success:
                print(f"Optimization Warning: {result.message}")
                
            return dict(zip(returns.columns, result.x))
            
        except Exception as e:
            print(f"Optimization Error: {str(e)}")
            return dict(zip(returns.columns, [1/n_assets] * n_assets))

# 最小方差优化策略
class MinVarianceOptimizationStrategy(OptimizationStrategy):
    def optimize(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
        """基于最小方差优化投资组合"""
        n_assets = returns.shape[1]
        
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        try:
            result = minimize(objective, 
                            x0=np.array([1/n_assets] * n_assets),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if not result.success:
                print(f"Optimization Warning: {result.message}")
                
            return dict(zip(returns.columns, result.x))
            
        except Exception as e:
            print(f"Optimization Error: {str(e)}")
            return dict(zip(returns.columns, [1/n_assets] * n_assets))

# 最大化收益优化策略
class MaxReturnOptimizationStrategy(OptimizationStrategy):
    def optimize(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
        """Optimize portfolio based on maximum return
        
        Args:
            returns: Returns data
            risk_free_rate: Risk-free rate (not used in this strategy)
            
        Returns:
            Dict: Dictionary of optimized asset weights
        """
        n_assets = returns.shape[1]
        
        def objective(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            return -portfolio_return  # Negative because we minimize
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        try:
            result = minimize(objective, 
                            x0=np.array([1/n_assets] * n_assets),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if not result.success:
                print(f"Optimization Warning: {result.message}")
                
            return dict(zip(returns.columns, result.x))
            
        except Exception as e:
            print(f"Optimization Error: {str(e)}")
            return dict(zip(returns.columns, [1/n_assets] * n_assets))

# 修改后的 PortfolioOptimizer 类
class PortfolioOptimizer:
    def __init__(self, 
                 market_env: MarketEnvironment, 
                 strategy: OptimizationStrategy = None,
                 config: PortfolioConfig = None,
                 monitor: PortfolioMonitor = None):
        self.market_env = market_env
        self.strategy = strategy or SharpeOptimizationStrategy()
        self.config = config or PortfolioConfig()
        self.monitor = monitor or PortfolioMonitor()
        self.logger = PortfolioLogger("PortfolioOptimizer")
        
    def optimize(self, window_days: int = 30, current_date: datetime = None) -> Dict:
        """Optimize portfolio weights
        
        Args:
            window_days: Number of days to use for optimization window
            current_date: Current date for optimization (for backtesting)
            
        Returns:
            Dict: Optimized weights for each asset
        """
        try:
            self.logger.logger.info(f"Starting portfolio optimization for date: {current_date}")
            start_time = time.time()
            
            # Get returns for the specified window
            returns = self.market_env.get_returns()
            if current_date is not None:
                # Get data up to current_date
                returns = returns[returns.index <= current_date]
                # Use only the last window_days
                returns = returns.tail(window_days)
            
            result = self.strategy.optimize(
                returns,
                self.config.get('risk_free_rate', 0.02)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.monitor.record_metric('optimization_duration', duration)
            self.logger.logger.info(f"Optimization completed in {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(str(e))

class BackTester:
    """回测框架"""
    
    def __init__(self, market_env: MarketEnvironment, optimizer: PortfolioOptimizer):
        self.market_env = market_env
        self.optimizer = optimizer
        
    def run_backtest(self, rebalance_frequency: str = '1M') -> pd.DataFrame:
        """执行回测并返回权重和收益的时间序列
        
        Args:
            rebalance_frequency: 再平衡频率 ('1D' 每日, '1W' 每周, '1M' 每月等)
            
        Returns:
            pd.DataFrame: 包含每日收益和资产权重的DataFrame
        """
        returns = self.market_env.get_returns()
        
        if returns.empty:
            raise ValueError("Returns data is empty. Please check if market data was correctly retrieved.")
            
        portfolio_data = []
        window_days = 30
        asset_tickers = [asset.ticker for asset in self.market_env.assets]

        start_date = returns.index[0] + pd.Timedelta(days=window_days)
        
        for date in pd.date_range(start=start_date, 
                                end=returns.index[-1], 
                                freq=rebalance_frequency):
            try:
                # Pass current_date to optimize method
                current_weights = self.optimizer.optimize(
                    window_days=window_days,
                    current_date=date
                )
                
                if date in returns.index:
                    daily_return = sum(returns.loc[date] * list(current_weights.values()))
                    
                    record = {
                        'Date': date,
                        'Returns': daily_return
                    }
                    for ticker, weight in current_weights.items():
                        record[f'{ticker}_weight'] = weight
                        
                    portfolio_data.append(record)
                    
            except Exception as e:
                print(f"Error occurred on date {date}: {str(e)}")
                continue
        
        results = pd.DataFrame(portfolio_data)
        results.set_index('Date', inplace=True)
        results = results.sort_index(ascending=True)
        
        return results

class PortfolioVisualizer:
    """Portfolio Visualization Tool"""
    
    def __init__(self, backtester: BackTester, strategy_name: str = "Default"):
        self.backtester = backtester
        self.strategy_name = strategy_name
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
        # Add strategy name to filename
        base_name, ext = os.path.splitext(filename)
        filename = f"{base_name}_{self.strategy_name.replace(' ', '_').lower()}{ext}"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
    
    def plot_cumulative_returns(self, results: pd.DataFrame) -> None:
        """Plot portfolio cumulative returns"""
        plt.figure(figsize=(12, 6))
        cumulative_returns = (1 + results['Returns']).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, label='Portfolio Cumulative Returns')
        plt.title(f'Portfolio Cumulative Performance - {self.strategy_name}')
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
        plt.title(f'Portfolio Asset Allocation - {self.strategy_name}')
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
        ax1.set_title(f'Rolling Annual Return - {self.strategy_name}')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(rolling_vol.index, rolling_vol.values, label='Annual Volatility', color='orange')
        ax2.set_title(f'Rolling Annual Volatility - {self.strategy_name}')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        self._save_plot('rolling_metrics.png')
        plt.close()

    def plot_weights_timeline(self, weights_history: Dict[str, pd.Series]) -> None:
        """Plot asset weights evolution over time"""
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
        
        plt.title(f'Portfolio Weights Evolution - {self.strategy_name}')
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
        """Plot asset weights heatmap over time"""
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
        
        plt.title(f'Portfolio Weights Heatmap - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Asset')
        
        plt.tight_layout()
        plt.show()
        self._save_plot('weights_heatmap.png')
        plt.close()

# Main program example
if __name__ == "__main__":
    try:
        # Initialize configuration and monitoring
        config = PortfolioConfig('C:/Users/zheng/.cursor-tutor/projects/python/portfolio_config.yaml')
        monitor = PortfolioMonitor()        
        
        # Get configuration values
        data_source = config.get('data.source')
        risk_free_rate = config.get('optimization.risk_free_rate')
        assets = config.get_assets()
        
        # Create asset list
        portfolio_assets = []
        for asset_config in assets:
            if asset_config['type'] == 'equity':
                asset = Equity(
                    ticker=asset_config['ticker'],
                    name=asset_config['name'],
                    sector=asset_config['sector']
                )
            elif asset_config['type'] == 'bond':
                asset = Bond(
                    ticker=asset_config['ticker'],
                    name=asset_config['name'],
                    duration=asset_config['duration']
                )
            portfolio_assets.append(asset)
            
        # Create market environment
        market_env = MarketEnvironment(
            assets=portfolio_assets,
            start_date=config.get('data.start_date'),
            end_date=config.get('data.end_date'),
            data_source=data_source,
            debug=config.get('debug.verbose', False)
        )
    
        # Create and test different optimization strategies
        strategies = {
            "Sharpe Ratio": SharpeOptimizationStrategy(),
            "Minimum Variance": MinVarianceOptimizationStrategy(),
            "Maximum Return": MaxReturnOptimizationStrategy()
        }

        # Initialize results dictionary to store performance metrics
        results_summary = {}

        # Test each strategy
        for strategy_name, strategy in strategies.items():
            print(f"\n{'='*20} Testing {strategy_name} Strategy {'='*20}")
            
            # Create optimizer with current strategy
            optimizer = PortfolioOptimizer(
                market_env=market_env,
                strategy=strategy,
                config=config,
                monitor=monitor
            )
            
            # Create backtester
            backtester = BackTester(market_env, optimizer)
            
            # Run backtest
            try:
                backtest_results = backtester.run_backtest(
                    rebalance_frequency=config.get('optimization.rebalance_frequency', '1M')
                )
                
                # Create visualizer with strategy name
                visualizer = PortfolioVisualizer(backtester, strategy_name)
                
                # Plot results
                visualizer.plot_cumulative_returns(backtest_results)
                
                # Get weight-related columns
                weight_cols = [col for col in backtest_results.columns if col.endswith('_weight')]
                weights_history = {col.replace('_weight', ''): backtest_results[col] 
                                 for col in weight_cols}
                
                # Plot weight timeline and heatmap
                visualizer.plot_weights_timeline(weights_history)
                visualizer.plot_weights_heatmap(weights_history)
                
                # Calculate performance metrics
                returns = backtest_results['Returns']
                cumulative_return = (1 + returns).cumprod().iloc[-1] - 1
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
                max_drawdown = (returns.cumprod() / returns.cumprod().cummax() - 1).min()
                
                # Store results
                results_summary[strategy_name] = {
                    'Cumulative Return': f"{cumulative_return:.2%}",
                    'Annual Return': f"{annual_return:.2%}",
                    'Annual Volatility': f"{annual_vol:.2%}",
                    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                    'Maximum Drawdown': f"{max_drawdown:.2%}"
                }
                
                # Print current strategy results
                print(f"\n=== Performance Metrics for {strategy_name} ===")
                for metric, value in results_summary[strategy_name].items():
                    print(f"{metric}: {value}")
                
                # Print final weights
                print(f"\n=== Final Portfolio Weights ({strategy_name}) ===")
                final_weights = backtest_results[weight_cols].iloc[-1]
                for asset, weight in final_weights.items():
                    print(f"{asset.replace('_weight', '')}: {weight:.2%}")
                
            except Exception as e:
                print(f"Error during backtest for {strategy_name}: {str(e)}")
                continue

        # Print comparative results
        print("\n=== Strategy Comparison ===")
        comparison_df = pd.DataFrame(results_summary).T
        print(comparison_df)
        
        # Save results to CSV
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_df.to_csv(f'{results_dir}/strategy_comparison_{timestamp}.csv')
        print(f"\nResults saved to: {results_dir}/strategy_comparison_{timestamp}.csv")
        
    except Exception as e:
        logging.error(f"Portfolio operation failed: {str(e)}")
        raise


