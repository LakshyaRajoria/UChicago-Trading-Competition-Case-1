#!/usr/bin/env python

from multiprocessing.sharedctypes import Value
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
# from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
# import statsmodels.tsa.holtwinters
from pmdarima.arima.utils import ndiffs
import math
import asyncio
import csv

CONTRACTS = ["LBSJ","LBSM", "LBSQ", "LBSV", "LBSZ"]
ORDER_SIZE = 20
SPREAD = 5

class Case1ExampleBot(UTCBot):
    '''
    An example bot for Case 1 of the 2022 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    '''

    async def handle_round_started(self):
        '''
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        '''
        self.rain = 1
        self.n = 0
        self.limit = {"LBSJ":84,"LBSM":126, "LBSQ":168, "LBSV":210, "LBSZ":252}
        self.carry = 0.5
        self.actual_fair = 0
        self.ratio = pd.read_csv('/Users/kevin/Documents/Work/NYU/Competitions/UChicago_Trading_Competition/Cases/Data/Case 1 Training Data/Ratio.csv')
        self.current_index = 0
        self.fairs = {}
        self.order_book = {}
        self.pos = {}
        self.order_ids = {}
        for month in CONTRACTS:
            self.order_ids[month+' bid'] = ''
            self.order_ids[month+' ask'] = ''

            self.fairs[month] = 330

            self.order_book[month] = {
                'Best Bid':{'Price':0,'Quantity':0},
                'Best Ask':{'Price':0,'Quantity':0}}

            self.pos[month] = 0

        asyncio.create_task(self.update_quotes())


    def futures(self,ratios_df, day, rain, contract):
        self.n = len(ratios_df)
        limit = {"LBSJ":84,"LBSM":126, "LBSQ":168, "LBSV":210, "LBSZ":252}
        model = ARIMA(ratios_df['Daily Ratios'], order=(1,1,1)).fit()
        spot_price = model.forecast(1, alpha=0.05) * rain
        return (spot_price+self.carry)*(limit[contract] - day)*(math.e)**(0.0025*(limit[contract] - day)/(252))

    def ratio_recalc(self, actual_fair, index):
        actual_ratio = [(actual_fair - self.carry*(self.limit["LBSZ"] - index - 1)*(math.e)**(0.0025*(self.limit["LBSZ"] - index - 1)/(252)))/(self.rain*(math.e)**(0.0025*(self.limit["LBSZ"] - index - 1)/(252)))]
        return actual_ratio


    # fair value update
    def update_fairs(self, current_rain):
        for month in CONTRACTS:
            self.fairs[month] = self.futures(self.ratio, self.current_index, current_rain, month)

    async def update_quotes(self):
        '''
        This function updates the quotes at each time step. In this sample implementation we
        are always quoting symetrically about our predicted fair prices, without consideration
        for our current positions. We don't reccomend that you do this for the actual competition.
        '''
        while True:

            self.update_fairs(self.rain)

            for contract in CONTRACTS:
                best_bid_error = False
                best_ask_error = False

                try:
                    best_bid = self.order_book[contract]['Best Bid']['Price']
                    bid_qty = self.order_book[contract]['Best Bid']['Quantity']
                except:
                    best_bid_error = True

                try:
                    best_ask = self.order_book[contract]['Best Ask']['Price']
                    ask_qty = self.order_book[contract]['Best Ask']['Quantity']
                except:
                    best_ask_error = True

                if best_ask_error and not best_bid_error:
                    best_ask = best_bid
                    best_ask += 0.1
                if best_bid_error and not best_ask_error:
                    best_bid = best_ask
                    best_bid -= 0.1
                if best_bid_error and best_ask_error:
                    continue

                self.actual_fair = (best_ask + best_bid)/2

                val_to_append = self.ratio_recalc(self.actual_fair, self.n)

                self.ratio.loc[len(self.ratio)] = val_to_append
                #self.ratio_recalc(actual_fair=self.actual_fair, index= self.n)
                # print(self.ratio)

                market_spread = float(best_ask) - float(best_bid)
                if market_spread < 0:
					#execute on free EV
					# to do this we place an ask equal to their bid and a bid equal to their ask, size it to 1/2 bc we wanna get filled fast
                    await self.place_bid_update(contract, int(float(ask_qty))/2, float(best_ask), None)
                    await self.place_ask_update(contract, int(float(bid_qty))/2, float(best_bid), None)
                    continue # We wanna wait
                size = self.pos[contract]
                fade_severity = 4 # adjust on the fly if we think we are underinformed (move up) / overinformed (move down)
                fade_amt = (-0.1)*((size/fade_severity)**1.3)
                self.fairs[contract] += fade_amt

                slack_size = 2  #Adjust on the fly (adjust down if we need to get filled more/move more size)
                slack = market_spread/slack_size
                ideal_bid = self.fairs[contract] - slack
                ideal_ask = self.fairs[contract] + slack #pennying
                if float(ideal_bid) <= float(best_bid):
                    ideal_bid = float(best_bid)+0.1
                if float(ideal_ask) >= float(best_ask):
                    ideal_ask = float(best_ask)-0.1

                current_spread = ideal_ask - ideal_bid

                if float(current_spread) <=0:
                    ideal_bid = float(best_bid)
                    ideal_ask = float(best_ask)

                ideal_bid = (ideal_bid)
                ideal_ask = (ideal_ask)

                bid_response = await self.modify_order(
                    self.order_ids[contract+' bid'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    ORDER_SIZE,
                    ideal_bid)

                ask_response = await self.modify_order(
                    self.order_ids[contract+' ask'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    ORDER_SIZE,
                    ideal_ask)

                assert bid_response.ok
                self.order_ids[contract+' bid'] = bid_response.order_id

                assert ask_response.ok
                self.order_ids[contract+' ask'] = ask_response.order_id

                if self.pos[contract] + ORDER_SIZE > 100:
                    ideal_bid = None
                    if self.order_ids[contract+' bid'] != None:
                        self.cancel_order(self.order_ids[contract+' bid'])
                if self.pos[contract] - ORDER_SIZE < -100:
                    ideal_ask = None
                    if self.order_ids[contract+' ask'] != None:
                        self.cancel_order(self.order_ids[contract+' ask'])



            await asyncio.sleep(1)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        This function receives messages from the exchange. You are encouraged to read through
        the documentation for the exachange to understand what types of messages you may receive
        from the exchange and how they may be useful to you.

        Note that monthly rainfall predictions are sent through Generic Message.
        '''
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            print('Realized pnl:', update.pnl_msg.realized_pnl)
            print("M2M pnl:", update.pnl_msg.m2m_pnl)

        elif kind == "market_snapshot_msg":
        # Updates your record of the Best Bids and Best Asks in the market
            self.current_index +=1
            for contract in CONTRACTS:
                book = update.market_snapshot_msg.books[contract]
                if len(book.bids) != 0:
                    best_bid = book.bids[0]
                    self.order_book[contract]['Best Bid']['Price'] = float(best_bid.px)
                    self.order_book[contract]['Best Bid']['Quantity'] = best_bid.qty

                if len(book.asks) != 0:
                    best_ask = book.asks[0]
                    self.order_book[contract]['Best Ask']['Price'] = float(best_ask.px)
                    self.order_book[contract]['Best Ask']['Quantity'] = best_ask.qty

        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.pos[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.pos[fill_msg.asset] -= update.fill_msg.filled_qty

        elif kind == "generic_msg":
            # Saves the predicted rainfall
            try:
                pred = float(update.generic_msg.message)
                self.rain = pred
            # Prints the Risk Limit message
            except ValueError:
                print(update.generic_msg.message)


if __name__ == "__main__":
    start_bot(Case1ExampleBot)
