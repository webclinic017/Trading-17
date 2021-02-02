import importlib.util
from backtesting import Backtest

from backtesting.test import GOOG

def main_AI(Strategy_file, Data_file):
    # Read the bot stratergy
    spec = importlib.util.spec_from_file_location('bot', Strategy_file)
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)

    # Spawn an empty version of that bot
    bot = bot.Bot()

    bot.train(GOOG)

